const DATE_TYPE_PATTERN = /DATE|TIME/i;

const compact = value => String(value ?? '').replaceAll('_', ' ').replace(/\s+/g, ' ').trim();
const unique = values => [...new Set(values.filter(Boolean))];

export function buildColumnDescription(column, profile = {}) {
  const parts = [
    `Column ${compact(column.name)}.`,
    `DuckDB type ${column.type}.`,
  ];
  if (profile.role) parts.push(`${profile.role}.`);
  if (Number.isFinite(profile.missingPercent)) parts.push(`${profile.missingPercent.toFixed(1)} percent missing.`);
  if (Number.isFinite(profile.cardinality)) parts.push(`${profile.cardinality} distinct values.`);
  if (profile.examples?.length) parts.push(`Example values: ${profile.examples.map(compact).join(', ')}.`);
  return parts.join(' ');
}

export function inferDatasetQuestions({ columns = [], numeric = [], profiles = {}, sourceName = 'uploaded data' } = {}) {
  const names = columns.map(column => column.name);
  const numericNames = numeric.filter(name => names.includes(name));
  const date = columns.find(column => DATE_TYPE_PATTERN.test(column.type))?.name;
  const categorical = columns.find(column => {
    if (numericNames.includes(column.name) || DATE_TYPE_PATTERN.test(column.type)) return false;
    const cardinality = profiles[column.name]?.cardinality;
    return !Number.isFinite(cardinality) || cardinality <= 50;
  })?.name;
  const [firstNumeric, secondNumeric] = numericNames;
  const missing = columns.filter(column => (profiles[column.name]?.missingPercent || 0) > 0).map(column => column.name);
  const dataset = compact(sourceName.replace(/\.csv$/i, '')) || 'uploaded data';

  const semantic = [];
  const analyst = [];

  if (date && firstNumeric) {
    semantic.push(`Which columns can help explain changes in ${compact(firstNumeric)} over ${compact(date)}?`);
    analyst.push(`How does ${compact(firstNumeric)} change over ${compact(date)}, and are there unusual periods?`);
  }
  if (categorical && firstNumeric) {
    semantic.push(`Which fields could explain differences in ${compact(firstNumeric)} across ${compact(categorical)}?`);
    analyst.push(`Which ${compact(categorical)} groups have the highest and lowest average ${compact(firstNumeric)}?`);
  }
  if (firstNumeric && secondNumeric) {
    semantic.push(`Which fields are relevant to the relationship between ${compact(firstNumeric)} and ${compact(secondNumeric)}?`);
    analyst.push(`What is the relationship between ${compact(firstNumeric)} and ${compact(secondNumeric)}, including correlation and outliers?`);
  }
  if (firstNumeric) {
    semantic.push(`Which columns may help explain unusually high or low ${compact(firstNumeric)} values?`);
    analyst.push(`What are the typical value, spread, distribution, and outliers for ${compact(firstNumeric)}?`);
  }
  if (missing.length) {
    semantic.push(`Which columns are most relevant to understanding missing values in ${missing.slice(0, 3).map(compact).join(', ')}?`);
    analyst.push(`How could missing values in ${missing.slice(0, 3).map(compact).join(', ')} affect the analysis?`);
  }
  semantic.push(`Which columns are most useful for investigating the main patterns in the ${dataset} dataset?`);
  analyst.push(`What are the strongest evidence-backed patterns in the ${dataset} dataset, and which SQL query should I run next?`);

  return {
    semantic: unique(semantic).slice(0, 4),
    analyst: unique(analyst).slice(0, 5),
  };
}
