import test from 'node:test';
import assert from 'node:assert/strict';
import {mean,variance,standardDeviation,pearson,linearRegression,iqrBounds,outliers,histogram,cosineSimilarity} from '../lib/stats.js';
const close=(a,b,e=1e-10)=>assert.ok(Math.abs(a-b)<e,`${a} ≠ ${b}`);
test('descriptive statistics are correct',()=>{close(mean([1,2,3,4,5]),3);close(variance([1,2,3,4,5]),2.5);close(standardDeviation([1,2,3,4,5]),Math.sqrt(2.5))});
test('Pearson correlation handles perfect linear relationships',()=>{close(pearson([1,2,3,4],[2,4,6,8]),1);close(pearson([1,2,3,4],[8,6,4,2]),-1)});
test('OLS recovers a known line',()=>{const r=linearRegression([1,2,3,4],[5,8,11,14]);close(r.slope,3);close(r.intercept,2);close(r.r2,1)});
test('IQR fences identify a clear extreme',()=>{const b=iqrBounds([1,2,2,3,3,4,100]);assert.ok(b.upper<100);assert.deepEqual(outliers([1,2,2,3,3,4,100]),[100])});
test('histogram conserves observations and cosine similarity is bounded',()=>{const h=histogram([1,2,3,4,5,6],3);assert.equal(h.reduce((s,b)=>s+b.count,0),6);close(cosineSimilarity([1,0],[1,0]),1);close(cosineSimilarity([1,0],[0,1]),0)});
