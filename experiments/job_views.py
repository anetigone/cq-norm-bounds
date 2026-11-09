"""Deterministic set-CQ derivation and exact SQL evaluation for JOBjoin."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
import re


ATOM = r"([a-z_][a-z_0-9]*)\s+AS\s+([a-z_][a-z_0-9]*)"
COLUMN = r"[a-z_][a-z_0-9]*\.[a-z_][a-z_0-9]*"
CATEGORICAL_COLUMNS = ('company_type_id', 'info_type_id', 'kind_id', 'role_id', 'link_type_id',
                       'subject_id', 'status_id', 'production_year')


@dataclass(frozen=True)
class Atom:
    table: str
    alias: str
    columns: tuple[tuple[str, str], ...]  # (source column, CQ variable)

    @property
    def scope(self):
        return frozenset(variable for _, variable in self.columns)

    def sql(self, key_filter=None, output=None, sample=None):
        selectors = {}
        predicates = []
        for column, variable in self.columns:
            if variable in selectors:
                predicates.append(f'{column} = {selectors[variable]}')
            else:
                selectors[variable] = column
            if variable.startswith('v'):
                predicates.append(f'{column} IS NOT NULL')
            if key_filter == variable:
                predicates.append(f'{column} BETWEEN 1 AND 64')
        if sample is not None:
            keys, values = sample
            predicates.append(sample_predicate(selectors, keys, values))
        projection = ', '.join(f'{selectors[variable]} AS {variable}' for variable in (output or selectors))
        where = ' WHERE ' + ' AND '.join(predicates) if predicates else ''
        return f'SELECT DISTINCT {projection} FROM {self.table}{where}'


@dataclass(frozen=True)
class Split:
    atoms: tuple[Atom, ...]
    left: tuple[Atom, ...]
    right: tuple[Atom, ...]
    keys: tuple[str, ...]
    left_head: tuple[str, ...]
    right_head: tuple[str, ...]

    @property
    def head(self):
        return tuple(sorted(set(self.left_head) | set(self.right_head)))


def parse_jobjoin(sql):
    """Accept only the source workload's qualified, conjunctive equijoins."""
    match = re.fullmatch(r'SELECT\s+\*\s+FROM\s+(.+?)\s+WHERE\s+(.+);', sql.strip(), re.I)
    if not match:
        raise ValueError(f'Unsupported JOBjoin statement: {sql}')
    tables = {}
    for atom in match[1].split(','):
        parsed = re.fullmatch(ATOM, atom.strip(), re.I)
        if parsed is None or parsed[2] in tables:
            raise ValueError(f'Unsupported table reference: {atom}')
        tables[parsed[2]] = parsed[1]

    parent = {}

    def root(column):
        while parent[column] != column:
            column = parent[column]
        return column

    for predicate in re.split(r'\s+AND\s+', match[2], flags=re.I):
        equality = re.fullmatch(rf'({COLUMN})\s*=\s*({COLUMN})', predicate.strip())
        if equality is None:
            raise ValueError(f'Unsupported JOBjoin predicate: {predicate}')
        left, right = equality.groups()
        for column in (left, right):
            parent.setdefault(column, column)
        parent[root(right)] = root(left)
    representatives = {representative: f'v{i:02d}' for i, representative in
                       enumerate(sorted({root(column) for column in parent}))}
    columns = defaultdict(list)
    for qualified in sorted(parent):
        alias, column = qualified.split('.')
        if alias not in tables:
            raise ValueError(f'Unknown alias: {alias}')
        columns[alias].append((column, representatives[root(qualified)]))
    return tuple(Atom(table, alias, tuple(columns[alias])) for alias, table in tables.items())


def scope(atoms):
    return frozenset().union(*(atom.scope for atom in atoms))


def connected(atoms):
    remaining = list(atoms[1:])
    seen = set(atoms[0].scope)
    while remaining:
        adjacent = [atom for atom in remaining if seen & atom.scope]
        if not adjacent:
            return False
        for atom in adjacent:
            seen.update(atom.scope)
            remaining.remove(atom)
    return True


def derive(atoms, table_rows, view_atoms=2, categorical_columns=None, *, categorical_priority=CATEGORICAL_COLUMNS):
    """Select a connected view with a local witness and a connected complement."""
    candidates = []
    for left in combinations(atoms, view_atoms):
        if not connected(left):
            continue
        right = tuple(atom for atom in atoms if atom not in left)
        if not right or not connected(right):
            continue
        left_scope, right_scope = scope(left), scope(right)
        keys = left_scope & right_scope
        if not keys or not left_scope - keys:
            continue
        # Prefer a small boundary, then a cheap inner join by base row counts.
        sizes = sorted(table_rows[atom.table] for atom in left)
        score = (len(keys), *sizes, tuple(atom.alias for atom in left))
        candidates.append((score, left, right, tuple(sorted(keys))))
    if not candidates:
        return None
    def add_payload(side, atom, column):
        variable = f'p_{atom.alias}'
        extended = Atom(atom.table, atom.alias, atom.columns + ((column, variable),))
        return tuple(extended if member == atom else member for member in side), variable

    def payload(side, boundary, preserve_witness):
        local = scope(side) - set(boundary)
        if preserve_witness and categorical_columns is not None:
            for column in categorical_priority:
                for atom in sorted(side, key=lambda item: (table_rows[item.table], item.alias)):
                    if column not in categorical_columns[atom.table]:
                        continue
                    variable = dict(atom.columns).get(column)
                    if variable is None:
                        return add_payload(side, atom, column)
                    if variable in local and len(local) > 1:
                        return side, variable
            return side, None
        if len(local) > int(preserve_witness):
            return side, min(local)
        # Use an existing JOB primary key as a payload when retaining a join
        # variable here would leave no internal witness to project away.
        eligible = [atom for atom in side if all(column != 'id' for column, _ in atom.columns)]
        if not eligible:
            return side, None
        chosen = min(eligible, key=lambda atom: (table_rows[atom.table], atom.alias))
        return add_payload(side, chosen, 'id')

    for _, left, right, keys in sorted(candidates):
        left, left_payload = payload(left, keys, preserve_witness=True)
        if left_payload is None:
            continue
        right, right_payload = payload(right, keys, preserve_witness=False)
        left_head = tuple(sorted((*keys, left_payload)))
        right_head = tuple(sorted((*keys, right_payload))) if right_payload else keys
        by_alias = {atom.alias: atom for atom in left + right}
        return Split(tuple(by_alias[atom.alias] for atom in atoms), left, right, keys, left_head, right_head)
    return None


def projection(factors, output):
    remaining = list(factors)
    sources = remaining.pop(0)
    seen = set(factors[sources])
    while remaining:
        adjacent = next((name for name in remaining if seen & factors[name]), None)
        name = adjacent or remaining[0]
        sources += (' NATURAL JOIN ' if adjacent else ' CROSS JOIN ') + name
        remaining.remove(name)
        seen.update(factors[name])
    return f'SELECT DISTINCT {", ".join(output)} FROM {sources}'


def project_join(atoms, head):
    """Exact variable elimination with DISTINCT after each witness projection."""
    factors = {f'r{i}': atom.scope for i, atom in enumerate(atoms)}
    ctes = [f'r{i} AS ({atom.sql()})' for i, atom in enumerate(atoms)]
    hidden = scope(atoms) - set(head)
    for index in range(len(hidden)):
        def elimination(variable):
            names = [name for name, variables in factors.items() if variable in variables]
            union = frozenset().union(*(factors[name] for name in names))
            return (len(union), len(names), variable), names, union

        (_, _, variable), inputs, merged = min(elimination(variable) for variable in hidden)
        output = merged - {variable}
        if not output:
            raise ValueError('Elimination produced a disconnected zero-column factor')
        hidden -= {variable}
        input_factors = {name: factors.pop(name) for name in inputs}
        name = f'e{index}'
        factors[name] = output
        ctes.append(f'{name} AS ({projection(input_factors, sorted(output))})')
    return f'WITH {", ".join(ctes)} {projection(factors, head)}'


def sample_predicate(columns, keys, values):
    if not set(keys) & columns.keys():
        return 'TRUE'
    terms = {tuple(f'{columns[key]} = {int(value)}' for key, value in zip(keys, row) if key in columns)
             for row in values}
    return '(' + ' OR '.join('(' + ' AND '.join(term) + ')' for term in sorted(terms)) + ')'


def check_chain_sql(split, source_sql, stages, tail, matched_keys=None):
    """Compare a view chain with flat SQL using the original join predicates."""
    sample = (split.keys, matched_keys) if matched_keys else None
    ctes = [f'a_{atom.alias} AS ({atom.sql(None if sample else split.keys[0], sample=sample)})'
            for atom in split.atoms]
    physical = {variable: f'{atom.alias}.{column}' for atom in split.atoms for column, variable in atom.columns}
    selectors = ', '.join(f'{physical[variable]} AS {variable}' for variable in split.head)
    flat = re.sub(r'^\s*SELECT\s+\*', f'SELECT DISTINCT {selectors}', source_sql, count=1, flags=re.I)
    flat_predicate = sample_predicate(physical, *sample) if sample else f'{physical[split.keys[0]]} BETWEEN 1 AND 64'
    flat = flat.rstrip().removesuffix(';') + f' AND {flat_predicate}'
    ctes.append(f'flat AS ({flat})')
    inputs = {}
    for index, (atoms, head) in enumerate(stages):
        inputs.update((f'a_{atom.alias}', atom.scope) for atom in atoms)
        name = f'view_{index}'
        ctes.append(f'{name} AS ({projection(inputs, head)})')
        inputs = {name: frozenset(head)}
    inputs.update((f'a_{atom.alias}', atom.scope) for atom in tail)
    nested = projection(inputs, split.head)
    if sample:
        nested = f'SELECT * FROM ({nested}) WHERE {sample_predicate(dict(zip(split.head, split.head)), *sample)}'
    ctes.append(f'nested AS ({nested})')
    return (f'WITH {", ".join(ctes)} '
            'SELECT (SELECT count(*) FROM flat), (SELECT count(*) FROM '
            '((SELECT * FROM flat EXCEPT SELECT * FROM nested) UNION ALL '
            '(SELECT * FROM nested EXCEPT SELECT * FROM flat)))')


def check_split_sql(split, source_sql, retain_witnesses):
    head = sorted(scope(split.left)) if retain_witnesses else split.left_head
    return check_chain_sql(split, source_sql, [(split.left, head)], split.right)


def degree_counts_sql(sql, key):
    if not key:
        return f'SELECT count(*) AS degree FROM ({sql}) HAVING count(*) > 0'
    return f'SELECT {", ".join(key)}, count(*) AS degree FROM ({sql}) GROUP BY {", ".join(key)}'


def histogram_sql(sql, key):
    return f'SELECT degree, count(*) FROM ({degree_counts_sql(sql, key)}) GROUP BY degree'


def joined_degree_counts_sql(left_sql, right_sql, left_head, right_head, join_key, key):
    """Count distinct factorized output degrees without enumerating the product."""
    left_key = sorted((set(key) & set(left_head)) | set(join_key))
    right_key = sorted((set(key) & set(right_head)) | set(join_key))
    left = degree_counts_sql(left_sql, left_key)
    right = degree_counts_sql(right_sql, right_key)
    fields = ', '.join(key)
    product = 'sum(l.degree::HUGEINT * r.degree) AS degree'
    grouping = f' GROUP BY {fields}' if key else ' HAVING count(*) > 0'
    return f'SELECT {fields + ", " if key else ""}{product} FROM ({left}) l JOIN ({right}) r USING ({", ".join(join_key)}){grouping}'


def joined_histogram_sql(left_sql, right_sql, left_head, right_head, join_key, key):
    degrees = joined_degree_counts_sql(left_sql, right_sql, left_head, right_head, join_key, key)
    return f'SELECT degree, count(*) FROM ({degrees}) GROUP BY degree'


def outer_cardinality_sql(left_sql, right_sql, keys):
    left_counts, right_counts = (degree_counts_sql(sql, keys) for sql in (left_sql, right_sql))
    return (f'SELECT coalesce(sum(l.degree::HUGEINT * r.degree), 0) '
            f'FROM ({left_counts}) l JOIN ({right_counts}) r USING ({", ".join(keys)})')


def degree_log(histogram, q):
    if not histogram:
        return -math.inf
    if q == math.inf:
        return math.log2(max(degree for degree, _ in histogram))
    moment = sum(count * degree**q for degree, count in histogram)
    return math.log2(moment) / q


def imdb_identity(tables):
    return {name: {'bytes': row['bytes'], 'rows': row['rows']} for name, row in tables.items()}


def same_imdb_input(archived, tables):
    if 'data_identity' in archived:
        return archived['data_identity'] == imdb_identity(tables)
    return archived.get('data_sha256') == {name: row.get('sha256') for name, row in tables.items()}
