import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import lzma
import pickle

def SE(m,Delta,gamma2,mu2):
  x = m*m/Delta + gamma2
  return (x+ mu2)/(x+1)

def logZ(m,Delta,gamma2,mu2): #VALID ONLY FOR MU=0!!
  x = m*m/Delta + gamma2
  return 0.5*x + 0.5*np.log(2*np.pi/(1+x)) - m*m*m/(3*Delta)

def uninf_SE(gmin,gmax,Ng,mu,Doverrho2, max_iter=10000, tol=1e-6):
    Delta = Doverrho2
    m_init = 1e-6
    m_tab = np.zeros(Ng)
    it_tab = np.zeros(Ng)
    logZ_tab = np.zeros(Ng)
    for i_g,gamma2 in enumerate(np.linspace(gmin**2,gmax**2,Ng)):
        m_SE = m_init
        for i in range(max_iter):
            m_old = m_SE
            m_SE = 0.5*SE(m_SE,Delta,gamma2,mu) + 0.5 * m_SE
            if (np.abs(m_SE-m_old)/m_SE<tol):
                break
            if  np.abs(m_SE)<1e-50:
                m_SE=0
                break
        m_tab[i_g] = m_SE
        logZ_tab[i_g] = logZ(m_SE,Delta,gamma2,mu)
        it_tab[i_g] = i
    return m_tab,it_tab,logZ_tab

def inf_SE(gmin,gmax,Ng,mu,Doverrho2, max_iter=100000, tol=1e-12):
    Delta = Doverrho2
    m_init = 1.0
    m_tab = np.zeros(Ng)
    it_tab = np.zeros(Ng)
    logZ_tab = np.zeros(Ng)
    for i_g,gamma2 in enumerate(np.linspace(gmin**2,gmax**2,Ng)):
        m_SE = m_init
        for i in range(max_iter):
            m_old = m_SE
            m_SE = 0.5*SE(m_SE,Delta,gamma2,mu) + 0.5 * m_SE
            if (np.abs(m_SE-m_old)/m_SE<tol):
                break
            if  np.abs(m_SE)<1e-50:
                m_SE=0
                break
        m_tab[i_g] = m_SE
        logZ_tab[i_g] = logZ(m_SE,Delta,gamma2,mu)
        it_tab[i_g] = i
    return m_tab,it_tab,logZ_tab

def main():
    parser = argparse.ArgumentParser(description="Sample")
    #parser.add_argument("--N", type=int, default=10000, help="size")
    parser.add_argument("--Doverrho2", type=float, default=2.0, help="Delta over rho squared")
    parser.add_argument("--gmin", type=float, default=0.01, help="gmin")
    parser.add_argument("--gmax", type=float, default=0.01, help="gmax")
    parser.add_argument("--Ng", type=int, default=10, help="Ng")
    parser.add_argument("--mu", type=float, default=1e-6, help="mu")
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
    mu = args.mu
    save_dir = args.save_dir

    m_uninf,it_uninf,logZ_un = uninf_SE(gmin,gmax,Ng,mu,Doverrho2)
    m_inf,it_inf,logZ_in = inf_SE(gmin,gmax,Ng,mu,Doverrho2)
    #print("m_uninf = ", m_uninf, "it_uninf = ", it_uninf, "m_inf = ", m_inf, "it_inf = ", it_inf)
    diff = m_inf-m_uninf

    keys = [r"$\gamma^2$","step","# iter (uninf)","# iter (inf)", r'$m_{uninf}/\rho$', r'$m_{inf}/\rho$', r'$\delta m/\rho$', r'$\log Z_{uninf}$', r'$\log Z_{inf}$', r'$\Delta/\rho^2$', "gmin", "gmax", "mu", "Ng"]
    dict_list=[]
    for i,gamma2 in enumerate(np.linspace(gmin**2,gmax**2,Ng)):
        values = [gamma2,i,it_uninf[i],it_inf[i], m_uninf[i], m_inf[i], diff[i], logZ_un[i],logZ_in[i], Doverrho2, gmin, gmax, mu, Ng]
        dict_list.append(dict(zip(keys, values)))
    df = pd.DataFrame(dict_list)
    #dff = pd.DataFrame(values,keys)

    file_name = "DF_Diff_SparseT_Sph_SE_DoR2_" + str(Doverrho2) + "_gmin_" + str(gmin) + "_gmax_" + str(gmax) + "_Ng_" + str(Ng) + "_m_" + str(mu) + ".xz"
    with lzma.open(save_dir + file_name, "wb") as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()