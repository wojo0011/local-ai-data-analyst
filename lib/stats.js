export const cleanNumbers = values => {
  const result=[];
  for(const value of values){
    if(value==null||(typeof value==='string'&&!value.trim()))continue;
    const number=Number(value);
    if(Number.isFinite(number))result.push(number);
  }
  return result;
};
const cleanPairs = (xs,ys) => {
  const result=[];
  for(let index=0;index<xs.length;index++){
    const x=xs[index],y=ys[index];
    if(x==null||y==null||(typeof x==='string'&&!x.trim())||(typeof y==='string'&&!y.trim()))continue;
    const pair=[Number(x),Number(y)];
    if(pair.every(Number.isFinite))result.push(pair);
  }
  return result;
};
const quantileSorted=(sorted,p)=>{if(!sorted.length)return NaN;if(sorted.length===1)return sorted[0];const i=(sorted.length-1)*p,l=Math.floor(i),h=Math.ceil(i);return sorted[l]+(sorted[h]-sorted[l])*(i-l)};
const range = values => {
  if(!values.length)return {min:NaN,max:NaN};
  let min=values[0],max=values[0];
  for(let index=1;index<values.length;index++){const value=values[index];if(value<min)min=value;if(value>max)max=value}
  return {min,max};
};
export function mean(values){const x=cleanNumbers(values);return x.length?x.reduce((a,b)=>a+b,0)/x.length:NaN}
export function variance(values,sample=true){const x=cleanNumbers(values);if(x.length<(sample?2:1))return NaN;const m=mean(x);return x.reduce((s,v)=>s+(v-m)**2,0)/(x.length-(sample?1:0))}
export function standardDeviation(values,sample=true){const v=variance(values,sample);return Number.isFinite(v)?Math.sqrt(v):NaN}
export function quantile(values,p){return quantileSorted(cleanNumbers(values).sort((a,b)=>a-b),p)}
export const median=values=>quantile(values,.5);
export function iqrBounds(values){const sorted=cleanNumbers(values).sort((a,b)=>a-b),q1=quantileSorted(sorted,.25),q3=quantileSorted(sorted,.75),iqr=q3-q1;return {q1,q3,iqr,lower:q1-1.5*iqr,upper:q3+1.5*iqr}}
export function outliers(values){const x=cleanNumbers(values),b=iqrBounds(x);return x.filter(v=>v<b.lower||v>b.upper)}
export function covariance(xs,ys,sample=true){const pairs=cleanPairs(xs,ys);if(pairs.length<(sample?2:1))return NaN;const mx=mean(pairs.map(p=>p[0])),my=mean(pairs.map(p=>p[1]));return pairs.reduce((s,[x,y])=>s+(x-mx)*(y-my),0)/(pairs.length-(sample?1:0))}
export function pearson(xs,ys){const pairs=cleanPairs(xs,ys);if(pairs.length<2)return NaN;const a=pairs.map(p=>p[0]),b=pairs.map(p=>p[1]),sx=standardDeviation(a),sy=standardDeviation(b);if(!sx||!sy)return NaN;return covariance(a,b)/(sx*sy)}
export function linearRegression(xs,ys){const pairs=cleanPairs(xs,ys);if(pairs.length<2)return {slope:NaN,intercept:NaN,r2:NaN,n:pairs.length};const x=pairs.map(p=>p[0]),y=pairs.map(p=>p[1]),mx=mean(x),my=mean(y);const sxx=x.reduce((s,v)=>s+(v-mx)**2,0),sxy=pairs.reduce((s,[a,b])=>s+(a-mx)*(b-my),0);if(!sxx)return {slope:NaN,intercept:NaN,r2:NaN,n:pairs.length};const slope=sxy/sxx,intercept=my-slope*mx;const sst=y.reduce((s,v)=>s+(v-my)**2,0),sse=pairs.reduce((s,[a,b])=>s+(b-(intercept+slope*a))**2,0);return {slope,intercept,r2:sst?1-sse/sst:1,n:pairs.length}}
export function histogram(values,bins=12){const x=cleanNumbers(values);if(!x.length)return [];const {min,max}=range(x);if(min===max)return [{x0:min,x1:max,count:x.length}];const width=(max-min)/bins,result=Array.from({length:bins},(_,i)=>({x0:min+i*width,x1:i===bins-1?max:min+(i+1)*width,count:0}));x.forEach(v=>{const i=Math.min(bins-1,Math.floor((v-min)/width));result[i].count++});return result}
export function cosineSimilarity(a,b){const n=Math.min(a.length,b.length);let dot=0,aa=0,bb=0;for(let i=0;i<n;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i]}return aa&&bb?dot/(Math.sqrt(aa)*Math.sqrt(bb)):0}
export function describe(values){const x=cleanNumbers(values),sorted=[...x].sort((a,b)=>a-b),q1=quantileSorted(sorted,.25),q3=quantileSorted(sorted,.75),iqr=q3-q1,lower=q1-1.5*iqr,upper=q3+1.5*iqr,{min,max}=range(x);let flagged=0;for(const value of x)if(value<lower||value>upper)flagged++;return {n:x.length,mean:mean(x),median:quantileSorted(sorted,.5),sd:standardDeviation(x),min,max,q1,q3,iqr,lower,upper,outliers:flagged}}
