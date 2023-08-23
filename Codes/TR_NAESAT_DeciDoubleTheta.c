#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define K 5
#define N 100000
//#define THR -1
//#define h0 0.

#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)  // random float, uniform in (0,1)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)  // +1/-1 with same probability
//#define theta(x) ((x) > 0. ? 1 : 0)
//#define max(a,b) ((a) > (b) ? (a) : (b))

int M, c, q, flag_vec[K];
float CumTab[K], wTab0[K+1], wTab[K+1], *Q, *Qhat, htab[K-1], htabK[K], CumTabK[K+1], PhiRS, T;
/* variabili globali per il generatore random */
unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

void error(char const *string) {
  fprintf(stderr, "ERROR: %s\n", string);
  exit(EXIT_FAILURE);
}

unsigned randForInit(void) {
  unsigned long long y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7fffffff) + (y >> 31);
  if (myrand & 0x80000000) {
    myrand = (myrand & 0x7fffffff) + 1;
  }
  return myrand;
}

void initRandom(void) {
  int i;
  
  ip = 128;    
  ip1 = ip - 24;    
  ip2 = ip - 55;    
  ip3 = ip - 61;
  
  for (i = ip3; i < ip; i++) {
    ira[i] = randForInit();
  }
}

int OLDpoissRan(float d) { //NOTE: DON'T GO TO d TOO HIGH
  int k = 0;
  float L = exp(-d);
  float F = L;
  float q = FRANDOM;
  printf("# Prova0A0 F %f\n", F);
  
  while (q > F) {
    k++;
    L *= d/k;
    F += L;
    //printf("# Prova q %f   F %f\n",q, F);
  }
  printf("# Prova0A0\n");
  return k;
}

int factorial(int n)  
{  
  if (n == 0)  
    return 1;  
  else  
    return(n * factorial(n-1));  
}  

int binomial_coefficient(int n, int k) {
  // If k > n, the binomial coefficient is 0
  if (k > n) return 0;

  // If n = 0 or k = 0, the binomial coefficient is 1
  if (n == 0 || k == 0) return 1;

  // Recursively compute the binomial coefficient using Pascal's Triangle
  return binomial_coefficient(n-1, k-1) + binomial_coefficient(n-1, k);
}

int poissRan1(float d) { //SOURCE: https://www.johndcook.com/blog/2010/06/14/generating-poisson-random-values/
  float c = 0.767 - 3.36/d;
  float beta = M_PI/sqrt(3.0*d);
  float alpha = beta*d;
  float k = log(c) - d - log(beta);
  float lhs, rhs, u, x, v, y;
  int n;

  do {
    u = FRANDOM;
    x = (alpha - log((1.0 - u)/u))/beta;
    n = floor(x + 0.5);
    if (n < 0) continue;
    v = FRANDOM;
    y = alpha - beta*x;
    lhs = y + log(v/((1.0 + exp(y))*(1.0 + exp(y))));
    rhs = k + n*log(d) - log(factorial(n));
  } while (lhs > rhs);

  return n;
}

float gammln(float xx) {
    double x,y,tmp,ser;
    static double cof[6]={76.18009172947146,-86.50532032941677,
    24.01409824083091,-1.231739572450155,
    0.1208650973866179e-2,-0.5395239384953e-5};
    int j;
    y=x=xx;
    tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser=1.000000000190015;
    for (j=0;j<=5;j++) ser += cof[j]/++y;
    return -tmp+log(2.5066282746310005*ser/x);
}

int poissRan(float d) { //SOURCE: Numerical Recipes in C
  static float sq,alxm,g,oldm=(-1.0); 
  float em,t,y; 
  if (d < 12.0) { 
    if (d != oldm) { 
      oldm=d; 
      g=exp(-d); 
    } 
    em =-1; 
    t=1.0; 
    do { 
      ++em; 
      t *= FRANDOM; 
    } while (t > g); 
  } else { 
    if (d != oldm) { 
      oldm=d; 
      sq=sqrt(2.0*d); 
      alxm=log(d); 
      g=d*alxm-gammln(d+1.0); 
    } 
    do { 
      do { 
        y=tan(M_PI*FRANDOM);
        em=sq*y+d; 
      } while (em < 0.0); 
      em=floor(em);
      t=0.9*(1.0+y*y)*exp(em*alxm-gammln(em+1.0)-g);
    } while (FRANDOM > t); 
  } 
  return em; 
}

float gaussRan(void) {
  static int iset = 0;
  static float gset;
  float fac, rsq, v1, v2;
  
  if (iset == 0) {
    do {
      v1 = 2.0 * FRANDOM - 1.0;
      v2 = 2.0 * FRANDOM - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  } else {
    iset = 0;
    return gset;
  }
}

void ComputewT(float eps_brd, float eps_rcs)
{
  wTab0[0]=0.; //Different from zero in the case of higher temperatures
  wTab0[K]=0.; //Different from zero in the case of higher temperatures
  wTab[0]=0.; //Different from zero in the case of higher temperatures
  wTab[K]=0.; //Different from zero in the case of higher temperatures
  wTab0[1]=1. - eps_brd;
  wTab0[K-1]=1. - eps_brd; 
  wTab[1]=1. - eps_rcs;
  wTab[K-1]=1. - eps_rcs;
  for (int i = 2; i < K-1; i++) 
  {
    wTab0[i]=1.;
    wTab[i]=1.;
  }
}

void ComputeCT(void)
{
  int i;
  float norm, normK;
  
  CumTab[0]=wTab0[0];
  CumTabK[0]=wTab0[0];
  for (i = 1; i < K; i++)
  {
    CumTab[i] = CumTab[i-1] + wTab0[i]* binomial_coefficient(K-1,i);
    CumTabK[i] = CumTabK[i-1] + wTab0[i]* binomial_coefficient(K,i);
  }
  CumTabK[K] = CumTabK[K-1] + wTab0[K];
  norm = CumTab[K-1];
  normK = CumTabK[K];
  for (i = 0; i < K; i++)
  {
    CumTab[i] /= norm;
    CumTabK[i] /= normK;
  }
  CumTabK[K]=1.;
}

//void ComputePhiRS(float alpha)
//{
//  int i;
//  int two2K = 1 << K;
//  float sum = 0.0;
//  
//  for (i = 0; i < K+1; i++)
//  {
//    sum += binomial_coefficient(K,i) * wTab[i];
//  }
//  if (gamma_==0) PhiRS = (1-alpha*K)*log(2) + alpha*log(sum);
//  else PhiRS = (1-alpha*K)*log(2*cosh(hFP)) + alpha*log(sum);
//  //printf("RS: %f",PhiRS);
//
//}

unsigned pCondRan(void) { 
  int flag = 0;
  int i = 0;
  int j, r;
  float rnd;
  unsigned state = 0;
  for (j = 0; j < K-1; j++)
  {
    flag_vec[j]=0;
  }
  
  do 
  {
    rnd=FRANDOM;
  } while ((rnd <= 0.) || (rnd >=1.));
  while (flag==0)
  {
    if (rnd <= CumTab[i]) 
    {
      for (j = 0; j < i; j++)     
      {
        do
        {
          r = (int)(FRANDOM * (K-1));
        } while (flag_vec[r]==1);
        flag_vec[r]=1;
        state |= (1U << r);
      }
      flag=1;
    }
    i++;
  }
  return state; 
}

unsigned pRan(void) { 
  int flag = 0;
  int i = 0;
  int j, r;
  float rnd;
  unsigned state = 0;
  for (j = 0; j < K; j++)
  {
    flag_vec[j]=0;
  }
  
  do 
  {
    rnd=FRANDOM;
  } while ((rnd <= 0.) || (rnd >=1.));
  while (flag==0)
  {
    if (rnd <= CumTabK[i]) 
    {
      for (j = 0; j < i; j++)     
      {
        do
        {
          r = (int)(FRANDOM * (K));
        } while (flag_vec[r]==1);
        flag_vec[r]=1;
        state |= (1U << r);
      }
      flag=1;
    }
    i++;
  }
  return state; 
}

void initQrandom(int NT) {
  int i;
  
  for (i = 0; i < NT; i++) {
    Q[i] = 2*FRANDOM - 1; //1;
    Qhat[i] = 2*FRANDOM - 1; //This second line is superfluous
  }
  for (i = NT; i < N; i++) {
    Q[i] = 1;
    Qhat[i] = 1;
  }
}

void initQ(void) {
  int i;
  
  for (i = 0; i < N; i++) {
    Q[i] = 1; //FRANDOM; //1;
    Qhat[i] = 1; //This second line is superfluous
  }
}

float fFP(float *utab, int d) {
  float prod1=1.;
  float prod2=1.;
  int posNull = 0; 
  int negNull = 0;
  float h;

  for (int i = 0; i < d; i++)
  {
    prod1 *= (1+utab[i]);
    prod2 *= (1-utab[i]);
    if (utab[i] == -1.0) posNull++;
    if (utab[i] == 1.0) negNull++;
  } 
  if (prod1 +prod2 > 0.0) {
    h =(prod1-prod2)/(prod1+prod2);
  }
  else {
    if (posNull > negNull)
      h = -1.0;
    else if (posNull < negNull)
      h = 1.0;
    else
      h = 0.0;
  }
  return h;
}

//float Z0v(float *utab, int d) {
//  float prod1=1.;
//  float prod2=1.;
//  //int two2d = 1 << d;
//
//  for (int i = 0; i < d; i++)
//  {
//    prod1 *= (1+utab[i]);
//    prod2 *= (1-utab[i]);
//  } 
//  //printf("V: %f \n",  (prod1 + prod2));
//  return (exp(hFP)*prod1 + exp(-hFP)*prod2);  //NOTE: NOT NORMALIZED
//}

float g(float *htab) {
  int i, spin;
  int nMinus;
  unsigned s, si;
  unsigned s_max = 1 << K;
  float num = 0.;
  float denum = 0.;
  float prod;
  
  for (s = 0; s < s_max; s++)
  {
    prod=1.;
    nMinus=0;
    si=s;
    for (i = 0; i < K-1; i++)
    {
      spin = 1 -2*(si&1);
      prod *= (1+htab[i]*spin);
      if (spin==-1) nMinus++;
      si >>= 1;
    }
    spin = 1 -2*(si&1);
    if (spin==-1) nMinus++;
    num += wTab[nMinus]*spin*prod;
    denum += wTab[nMinus]*prod;
  }
  
  return num/denum;
}

void updateQ_FP(float c, float NT) {
  int i, j, bit, spin, d, r;
  unsigned state;
  float *utab;

  state = pRan();
  for (i = 0; i < NT; i++)
  {
    d = poissRan(c);
    utab = (float*)calloc(d, sizeof(float));
    for (j = 0; j < d; j++)
    {
      r = (int)(FRANDOM * N);
      utab[j] = Qhat[r];
    } 
    Q[i] = fFP(utab, d);
    free(utab);
    //printf("###Try%f %f\n", oldQ[i], Q[i]);
  }
}

void updateQhat(void) {
  int i, j, bit, spin, d, r;
  unsigned state;
  for (size_t i = 0; i < N; i++)
  {
    state = pCondRan();
    for (j = 0; j < K-1; j++)
    {
      r = (int)(FRANDOM * N);
      bit = (state >> j) & 1U;
      spin = 1 - bit*2; //0 becomes +1 and 1 becomes -1
      htab[j] = spin*Q[r];
    }
    Qhat[i] = g(htab);
  }
}

float compute_q1() {
  int i;
  float q1 = 0.;

  for (i = 0; i < N; i++)
  {
    q1 += Q[i]*Q[i];
  }

  return q1/(float)N;
}

float compute_ov() {
  int i;
  float q1 = 0.;

  for (i = 0; i < N; i++)
  {
    q1 += Q[i];
  }

  return q1/(float)N;
}

float Z0v(float *utab, int d) {
  float prod1=1.;
  float prod2=1.;
  //int two2d = 1 << d;

  for (int i = 0; i < d; i++)
  {
    prod1 *= (1+utab[i]);
    prod2 *= (1-utab[i]);
  } 
  //printf("V: %f \n",  (prod1 + prod2));
  return prod1 + prod2;  //NOTE: NOT NORMALIZED
}

float Z0v_deci(float *utab, int d) {
  float prod1=1.;
  float prod2=1.;
  //int two2d = 1 << d;

  for (int i = 0; i < d; i++)
    prod1 *= (1+utab[i]);
  //printf("V: %f \n",  (prod1 + prod2));
  return prod1;  //NOTE: NOT NORMALIZED
}

float Z0c(float *htabK) {
  int i, spin;
  int nMinus;
  unsigned s, si;
  unsigned s_max = 1 << K;
  float z = 0.;
  float prod;
  for (s = 0; s < s_max; s++)
  {
    prod=1.;
    nMinus=0;
    si=s;
    for (i = 0; i < K; i++)
    {
      spin = 1 -2*(si&1);
      prod *= (1+htabK[i]*spin);
      if (spin==-1) nMinus++;
      si >>= 1;
    }
    z += wTab[nMinus]*prod;
    //printf("prod: %f \n", prod);
    //printf("z: %f \n", z);
  }
  //if (z==0.) printf("C: %f %f %f %f %f\n", htabK[0], htabK[1], htabK[2], htabK[3], htabK[4]);
  return z; //NOTE: NOT NORMALIZED
}

float Z0e(float h, float u) { 
  //printf("E: %f \n", 0.25*((1+h)*(1+u)+(1-h)*(1-u)));
  return 0.25*((1+h)*(1+u)+(1-h)*(1-u));
}

float compute_logZ(float alpha, float c) {
  int i, d, r, bit, spin;
  float u, h, posProd, negProd;
  float *utab;
  unsigned state;
  int it;
  int nrep = N;

  float logZ = 0.;
  for (int it = 0; it < nrep; it++)
  {

    //Part 1
    d = poissRan(c);
    utab = (float*)calloc(d, sizeof(float));
    for (i = 0; i < d; i++)
    {
      r = (int)(FRANDOM * N);
      utab[i] = Qhat[r];
      //printf("utabb: %f \n", utab[i]);
    } 
    logZ += (log(Z0v(utab, d)) - d*log(2));

    //printf("Part 1: %f \n", log(Z0v(utab, d)) -alpha*K*log(2));
    //Part 2
    state = pRan();
    for (i = 0; i < K; i++)
    {
      r = (int)(FRANDOM * N);
      bit = (state >> i) & 1U;
      spin = 1 - bit*2; //0 becomes +1 and 1 becomes -1
      htabK[i] = spin*Q[r];
      //printf("htabb: %f \n", htabK[i]);
    }
    //printf("Part 2: %f \n", alpha * (log(Z0c(htabK)) - K*log(2)));
    logZ += alpha * (log(Z0c(htabK)) - K*log(2));
    //Part 3
    r = (int)(FRANDOM * N);
    u = Qhat[r];
    r = (int)(FRANDOM * N);
    h = Q[r];
    //printf("Part 3: %f \n", alpha * K * log(Z0e(h,u)));
    logZ -= alpha * K * log(Z0e(h,u));
    free(utab);
  }
  return logZ/nrep;
}

float compute_logZ_deci(float alpha, float c, int NT) {
  int i, d, r, bit, spin;
  float u, h, posProd, negProd;
  float *utab;
  unsigned state;
  int it;
  int nrep = N;

  float logZ = 0.;
  for (int it = 0; it < nrep; it++)
  {

    //Part 1
    if (it < NT) {
      d = poissRan(c);
      utab = (float*)calloc(d, sizeof(float));
      for (i = 0; i < d; i++)
      {
        r = (int)(FRANDOM * N);
        utab[i] = Qhat[r];
        //printf("utabb: %f \n", utab[i]);
      } 
      logZ += (log(Z0v(utab, d)) - d*log(2));
    }
    else {
      d = poissRan(c);
      utab = (float*)calloc(d, sizeof(float));
      for (i = 0; i < d; i++)
      {
        r = (int)(FRANDOM * N);
        utab[i] = Qhat[r];
        //printf("utabb: %f \n", utab[i]);
      } 
      logZ += (log(Z0v_deci(utab, d)) - d*log(2));
    }

    //printf("Part 1: %f \n", log(Z0v(utab, d)) -alpha*K*log(2));
    //Part 2
    state = pRan();
    for (i = 0; i < K; i++)
    {
      r = (int)(FRANDOM * N);
      bit = (state >> i) & 1U;
      spin = 1 - bit*2; //0 becomes +1 and 1 becomes -1
      htabK[i] = spin*Q[r];
      //printf("htabb: %f \n", htabK[i]);
    }
    //printf("Part 2: %f \n", alpha * (log(Z0c(htabK)) - K*log(2)));
    logZ += alpha * (log(Z0c(htabK)) - K*log(2));
    //Part 3
    r = (int)(FRANDOM * N);
    u = Qhat[r];
    r = (int)(FRANDOM * N);
    h = Q[r];
    //printf("Part 3: %f \n", alpha * K * log(Z0e(h,u)));
    logZ -= alpha * K * log(Z0e(h,u));
    free(utab);
  }
  return logZ/nrep;
}

int main(int argc, char *argv[]) {
  int is, it, start, maxIter, seed, itu, NT;
  float alpha, c, ov, eps_brd, eps_rcs, q1, q1_ave, ov_ave, q1u, ovu, logZ, logZu,logZ_ave, T, Tmin, Tmax, Tstep;
  char filename[128];// folder_name[20];

  if (argc != 11) {
    fprintf(stderr, "usage: %s folder_name eps_brd eps_rcs alpha Tmin Tmax Tstep start maxIter [seed]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  //folder_name = argv[1];
  eps_brd = atof(argv[2]);
  eps_rcs = atof(argv[3]);
  alpha = atof(argv[4]);
  c = alpha*K;
  Tmin = atof(argv[5]);
  Tmax = atof(argv[6]);
  Tstep = atof(argv[7]);
  start = atoi(argv[8]);
  //if (init != 0 && init != 1) error("init must be 0 or 1");
  maxIter = atoi(argv[9]);
  seed = atoi(argv[10]);
  myrand = (unsigned)seed;
  if (myrand == 2147483647)
    error("seed must be less than 2147483647");

  printf("## eps_brd = %.3f   eps_rcs = %.3f   K = %i   N = %i   alpha = %.3f  Tmin = %.3f  Tmax = %.3f  Tstep = %.3f   start = %i   maxIter = %i   seed = %i\n",
	 eps_brd, eps_rcs, K, N, alpha, Tmin, Tmax, Tstep, start, maxIter, seed);
  //printf("# 1:t  2:q1_t\n");
  //printf("# 1:t  2:ov_t  3:ovave_t\n");

  initRandom();
  ComputewT(eps_brd, eps_rcs);
  ComputeCT();

  Q = (float *)calloc(N, sizeof(float));
  Qhat = (float *)calloc(N, sizeof(float));  
  sprintf(filename, "%s/dataFP_EB%.3f_ER%.3f_K%i_a%.3f_Tm%.3f_TM%.3f_Ts%.3f_st%i_s%i.txt", argv[1], eps_brd, eps_rcs, K, alpha, Tmin, Tmax, Tstep, start, seed );
  FILE *data_file = fopen(filename, "w");
  if (data_file == NULL) error("Error: Unable to open file\n");
  fprintf(data_file, "## eps_brd = %.3f   eps_rcs = %.3f   K = %i   N = %i   alpha = %.3f  Tmin = %.3f  Tmax = %.3f  Tstep = %.3f\n",
	 eps_brd, eps_rcs, K, N, alpha, Tmin, Tmax, Tstep);
  fprintf(data_file, "## start = %i   maxIter = %i   seed = %i\n",
	 start, maxIter, seed);
  fprintf(data_file, "#0:theta  1:ov_0  2:q1_0  3:logZ_0  4:ov_1  5:q1_1  6:logZ_1  7:delta_ov  8:delta_q1  9:delta_logZ\n");
  for (T = Tmin; T <= Tmax; T += Tstep)
  {
    NT = (int)(N*(1-T));
    initQrandom(NT);
    updateQhat();
    q1_ave=0.;
    ov_ave=0.;
    logZ_ave=0.;
    for (it = 0; it < maxIter; it++)
    {
      q1 = compute_q1();
      ov = compute_ov();
      logZ = compute_logZ_deci(alpha,c,NT);
      //cxt = compute_complexity(alpha, c);
      //fp = compute_FP(alpha, c);
      if (it > start) {
        q1_ave = (q1_ave*(it-start-1) + q1)/(it-start);
        ov_ave = (ov_ave*(it-start-1) + ov)/(it-start);
        logZ_ave = (logZ_ave*(it-start-1) + logZ)/(it-start);
      }
      //printf("#%i %f %f\n", it, ov, ov_ave);
      //fprintf(data_file, "%i %f %f\n", it, ov, ov_ave);
      updateQ_FP(c,NT);
      updateQhat();
      //fflush(stdout);
      //if (q1 < THR) it=maxIter;
    }
    q1u = q1_ave;
    ovu = ov_ave;
    logZu = logZ_ave;
    itu = it;
    //printf("#%i %f %f\n", it, ov, ov_ave);
    initQ();
    updateQhat();
    q1_ave=0.;
    ov_ave=0.;
    logZ_ave=0.;
    for (it = 0; it < maxIter; it++)
    {
      q1 = compute_q1();
      ov = compute_ov();
      logZ = compute_logZ_deci(alpha,c,NT);
      //cxt = compute_complexity(alpha, c);
      //fp = compute_FP(alpha, c);
      if (it > start) {
        q1_ave = (q1_ave*(it-start-1) + q1)/(it-start);
        ov_ave = (ov_ave*(it-start-1) + ov)/(it-start);
        logZ_ave = (logZ_ave*(it-start-1) + logZ)/(it-start);
      }
      //printf("#%i %f %f\n", it, ov, ov_ave);
      //fprintf(data_file, "%i %f %f\n", it, ov, ov_ave);
      updateQ_FP(c,NT);
      updateQhat();
      //fflush(stdout);
      //if (q1 < THR) it=maxIter;
    }
    fprintf(data_file, "%f %f %f %f %f %f %f %f %f %f\n", T, ovu, q1u, logZu, ov_ave, q1_ave, logZ_ave, ov_ave-ovu, q1_ave-q1u, logZ_ave-logZu);
    fflush(data_file);  
  }

  free(Q);
  free(Qhat);
  fclose(data_file);
  return EXIT_SUCCESS;
}
