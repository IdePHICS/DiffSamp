import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import argparse
import lzma
import pickle

def SEterm(z,m,Delta,rho):
  x = m/Delta 
  f = np.tanh(x + np.sqrt(x)*z)/(rho + (1-rho)/(np.exp(-0.5*(x))*np.cosh(x + np.sqrt(x)*z)))
  return rho*rho*np.exp(-(z*z)/2)*f/np.sqrt(2*np.pi)

def logZterm1(z,m,Delta,rho):
  x = m/Delta
  f1 = np.log(1-rho + rho*np.exp(-0.5*(x))*np.cosh(x + np.sqrt(x)*z))
  return np.exp(-(z*z)/2)*f1/np.sqrt(2*np.pi)

def logZterm2(z,m,Delta,rho):
  x = m/Delta
  f2 = np.log(1-rho + rho*np.exp(-0.5*(x))*np.cosh(np.sqrt(x)*z))
  return np.exp(-(z*z)/2)*f2/np.sqrt(2*np.pi)

def SE(m,Delta,rho):
  fct = lambda x: SEterm(x,m,Delta,rho)
  return scipy.integrate.quad(fct, -10, 10)[0]

def logZ(m,Delta,rho,Theta):
    fct1 = lambda x: logZterm1(x,m,Delta,rho)
    fct2 = lambda x: logZterm2(x,m,Delta,rho)
    fct3 = 0.
    if ((rho!=0) and (rho!=1)): fct3 = rho*np.log(rho) + (1-rho)*np.log(1-rho)
    fct4 = 0.5*rho*m/Delta
    return (1-Theta)*(rho * scipy.integrate.quad(fct1, -10, 10)[0] + (1-rho) * scipy.integrate.quad(fct2, -10, 10)[0]) + Theta*(fct3+fct4) - m*m/(4*Delta)

def uninf_SE(Tmin,Tmax,NT,rho,Doverrho2, max_iter=100000, tol=1e-8):
    Delta = rho*rho*Doverrho2
    m_init = 1e-6
    m_tab = np.zeros(NT)
    it_tab = np.zeros(NT)
    logZ_tab = np.zeros(NT)
    for i_T,Theta in enumerate(np.linspace(Tmin,Tmax,NT)):
        m_SE = m_init
        for i in range(max_iter):
            m_old = m_SE
            m_SE = 0.5*(rho*Theta + (1-Theta)*SE(m_SE,Delta,rho)) + 0.5 * m_SE
            if (np.abs(m_SE-m_old)/m_SE<tol):
                break
            if  np.abs(m_SE)<1e-50:
                m_SE=0
                break
        m_tab[i_T] = m_SE/rho
        logZ_tab[i_T] = logZ(m_SE,Delta,rho,Theta)
        it_tab[i_T] = i
    return m_tab,it_tab,logZ_tab

def inf_SE(Tmin,Tmax,NT,rho,Doverrho2, max_iter=100000, tol=1e-8):
    Delta = rho*rho*Doverrho2
    m_init = rho
    m_tab = np.zeros(NT)
    it_tab = np.zeros(NT)
    logZ_tab = np.zeros(NT)
    for i_T,Theta in enumerate(np.linspace(Tmin,Tmax,NT)):
        m_SE = m_init
        for i in range(max_iter):
            m_old = m_SE
            m_SE = 0.5*(rho*Theta + (1-Theta)*SE(m_SE,Delta,rho)) + 0.5 * m_SE
            if (np.abs(m_SE-m_old)/m_SE<tol):
                break
            if  np.abs(m_SE)<1e-50:
                m_SE=0
                break
        m_tab[i_T] = m_SE/rho
        logZ_tab[i_T] = logZ(m_SE,Delta,rho,Theta)
        it_tab[i_T] = i
    return m_tab,it_tab,logZ_tab

def main():
    parser = argparse.ArgumentParser(description="Sample")
    #parser.add_argument("--N", type=int, default=10000, help="size")
    parser.add_argument("--Doverrho2", type=float, default=2.0, help="Delta over rho squared")
    parser.add_argument("--Tmin", type=float, default=0.01, help="Tmin")
    parser.add_argument("--Tmax", type=float, default=0.01, help="Tmax")
    parser.add_argument("--NT", type=int, default=10, help="NT")
    parser.add_argument("--rho", type=float, default=1e-6, help="rho")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/check/",
        help="saving directory for data frames",
    )

    args = parser.parse_args()
    print("arguments:")
    print(args)

    Doverrho2 = args.Doverrho2
    Tmin = args.Tmin
    Tmax = args.Tmax
    NT = args.NT
    rho = args.rho
    save_dir = args.save_dir
    beta = 1/(rho*np.sqrt(Doverrho2))

    m_uninf,it_uninf,logZ_un = uninf_SE(Tmin,Tmax,NT,rho,Doverrho2)
    m_inf,it_inf,logZ_in = inf_SE(Tmin,Tmax,NT,rho,Doverrho2)
    diff = m_inf-m_uninf
    diff_logZ = logZ_in-logZ_un

    keys = [r"$\theta$","step","# iter (uninf)","# iter (inf)", r'$m_{uninf}/\rho$', r'$m_{inf}/\rho$', r'$\delta m/\rho$', r'$\log Z_{uninf}$', r'$\log Z_{inf}$', r'$\delta logZ$', r'$\Delta/\rho^2$', r"$\beta$", "Tmin", "Tmax", "rho", "NT"]
    dict_list=[]
    for i_T,Theta in enumerate(np.linspace(Tmin,Tmax,NT)):
        values = [Theta,i_T,it_uninf[i_T],it_inf[i_T], m_uninf[i_T], m_inf[i_T], diff[i_T], logZ_un[i_T],logZ_in[i_T], diff_logZ[i_T], Doverrho2, beta, Tmin, Tmax, rho, NT]
        dict_list.append(dict(zip(keys, values)))
    df = pd.DataFrame(dict_list)
    #dff = pd.DataFrame(values,keys)

    file_name = "DF_Deci_SpSK_SE_DoR2_" + str(Doverrho2) + "_Tm_" + str(Tmin) + "_TM_" + str(Tmax) + "_NT_" + str(NT) + "_r_" + str(rho) + ".xz"
    with lzma.open(save_dir + file_name, "wb") as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()