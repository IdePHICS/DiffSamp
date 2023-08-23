import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import argparse
import lzma
import pickle

def SEterm(z,m,Delta,gamma2,rho):
  x = m/Delta + gamma2
  f = np.tanh(x + np.sqrt(x)*z)/(rho + (1-rho)/(np.exp(-0.5*(x))*np.cosh(x + np.sqrt(x)*z)))
  return rho*rho*np.exp(-(z*z)/2)*f/np.sqrt(2*3.14159265359)

def logZterm1(z,m,Delta,gamma2,rho):
  x = m/Delta + gamma2
  f1 = np.log(1-rho + rho*np.exp(-0.5*(x))*np.cosh(x + np.sqrt(x)*z))
  return np.exp(-(z*z)/2)*f1/np.sqrt(2*3.14159265359)

def logZterm2(z,m,Delta,gamma2,rho):
  x = m/Delta + gamma2
  f2 = np.log(1-rho + rho*np.exp(-0.5*(x))*np.cosh(np.sqrt(x)*z))
  return np.exp(-(z*z)/2)*f2/np.sqrt(2*3.14159265359)

def SE(m,Delta,gamma2,rho):
  fct = lambda x: SEterm(x,m,Delta,gamma2,rho)
  return scipy.integrate.quad(fct, -6, 6)[0]

def logZ(m,Delta,gamma2,rho):
    fct1 = lambda x: logZterm1(x,m,Delta,gamma2,rho)
    fct2 = lambda x: logZterm2(x,m,Delta,gamma2,rho)
    return rho * scipy.integrate.quad(fct1, -6, 6)[0] + (1-rho) * scipy.integrate.quad(fct2, -6, 6)[0] - m*m/(4*Delta)

def uninf_SE(gmin,gmax,Ng,rho,Doverrho2, max_iter=10000, tol=1e-6):
    Delta = rho*rho*Doverrho2
    m_init = 1e-6
    m_tab = np.zeros(Ng)
    it_tab = np.zeros(Ng)
    logZ_tab = np.zeros(Ng)
    for i_g,gamma2 in enumerate(np.linspace(gmin**2,gmax**2,Ng)):
        m_SE = m_init
        for i in range(max_iter):
            m_old = m_SE
            m_SE = 0.5*SE(m_SE,Delta,gamma2,rho) + 0.5 * m_SE
            if (np.abs(m_SE-m_old)/m_SE<tol):
                break
            if  np.abs(m_SE)<1e-50:
                m_SE=0
                break
        m_tab[i_g] = m_SE/rho
        it_tab[i_g] = i
        logZ_tab[i_g] = logZ(m_SE,Delta,gamma2,rho)
    return m_tab,it_tab,logZ_tab

def inf_SE(gmin,gmax,Ng,rho,Doverrho2, max_iter=10000, tol=1e-6):
    Delta = rho*rho*Doverrho2
    m_init = rho
    m_tab = np.zeros(Ng)
    it_tab = np.zeros(Ng)
    logZ_tab = np.zeros(Ng)
    for i_g,gamma2 in enumerate(np.linspace(gmin**2,gmax**2,Ng)):
        m_SE = m_init
        for i in range(max_iter):
            m_old = m_SE
            m_SE = 0.5*SE(m_SE,Delta,gamma2,rho) + 0.5 * m_SE
            if (np.abs(m_SE-m_old)/m_SE<tol):
                break
            if  np.abs(m_SE)<1e-50:
                m_SE=0
                break
        m_tab[i_g] = m_SE/rho
        logZ_tab[i_g] = logZ(m_SE,Delta,gamma2,rho)
        it_tab[i_g] = i
    return m_tab,it_tab,logZ_tab

def main():
    parser = argparse.ArgumentParser(description="Sample")
    #parser.add_argument("--N", type=int, default=10000, help="size")
    parser.add_argument("--Doverrho2", type=float, default=2.0, help="Delta over rho squared")
    parser.add_argument("--gmin", type=float, default=0.01, help="gmin")
    parser.add_argument("--gmax", type=float, default=0.01, help="gmax")
    parser.add_argument("--Ng", type=int, default=10, help="Ng")
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
    gmin = args.gmin
    gmax = args.gmax
    Ng = args.Ng
    rho = args.rho
    save_dir = args.save_dir
    beta = 1/(rho*np.sqrt(Doverrho2))

    m_uninf,it_uninf, logZ_un = uninf_SE(gmin,gmax,Ng,rho,Doverrho2)
    m_inf,it_inf, logZ_in = inf_SE(gmin,gmax,Ng,rho,Doverrho2)
    diff = m_inf-m_uninf
    diff_logZ = logZ_in-logZ_un

    keys = [r"$\gamma^2$","step","# iter (uninf)","# iter (inf)", r'$m_{uninf}/\rho$', r'$m_{inf}/\rho$', r'$\delta m/\rho$', r'$\log Z_{uninf}$', r'$\log Z_{inf}$', r'$\delta logZ$', r'$\Delta/\rho^2$', r"$\beta$", "gmin", "gmax", "rho", "Ng"]
    dict_list=[]
    for i,gamma2 in enumerate(np.linspace(gmin**2,gmax**2,Ng)):
        values = [gamma2,i,it_uninf[i],it_inf[i], m_uninf[i], m_inf[i], diff[i], logZ_un[i],logZ_in[i],diff_logZ[i], Doverrho2, beta, gmin, gmax, rho, Ng]
        dict_list.append(dict(zip(keys, values)))
    df = pd.DataFrame(dict_list)
    #dff = pd.DataFrame(values,keys)

    file_name = "DF_Diff_SpSK_SE_DoR2_" + str(Doverrho2) + "_gmin_" + str(gmin) + "_gmax_" + str(gmax) + "_Ng_" + str(Ng) + "_r_" + str(rho) + ".xz"
    with lzma.open(save_dir + file_name, "wb") as f:
        pickle.dump(df, f)
    
if __name__ == "__main__":
    main()