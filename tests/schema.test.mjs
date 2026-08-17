import test from 'node:test';
import assert from 'node:assert/strict';
import { buildColumnDescription, inferDatasetQuestions } from '../lib/schema.js';

const columns=[
  {name:'order_date',type:'DATE'},
  {name:'channel',type:'VARCHAR'},
  {name:'revenue',type:'DOUBLE'},
  {name:'returns',type:'BIGINT'},
];
const profiles={
  order_date:{cardinality:30,missingPercent:0,examples:['2026-01-01']},
  channel:{cardinality:3,missingPercent:0,examples:['Online','Store']},
  revenue:{cardinality:25,missingPercent:0,examples:['100','125']},
  returns:{cardinality:8,missingPercent:5,examples:['0','1']},
};

test('questions are inferred from the uploaded schema instead of hard-coded demo fields',()=>{
  const questions=inferDatasetQuestions({columns,numeric:['revenue','returns'],profiles,sourceName:'orders.csv'});
  assert.ok(questions.semantic.some(question=>question.includes('revenue')&&question.includes('order date')));
  assert.ok(questions.analyst.some(question=>question.includes('channel')&&question.includes('average revenue')));
  assert.ok(questions.analyst.some(question=>question.includes('revenue')&&question.includes('returns')));
  assert.ok(questions.analyst.every(question=>!question.includes('return rate')));
});

test('one numeric variable still yields useful univariate questions',()=>{
  const questions=inferDatasetQuestions({columns:columns.slice(0,3),numeric:['revenue'],profiles,sourceName:'orders.csv'});
  assert.ok(questions.analyst.some(question=>question.includes('spread')&&question.includes('revenue')));
});

test('semantic column descriptions include observed profile evidence',()=>{
  const description=buildColumnDescription(columns[1],profiles.channel);
  assert.match(description,/channel/i);
  assert.match(description,/Online, Store/);
  assert.match(description,/3 distinct values/);
});
