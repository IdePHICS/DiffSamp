import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import argparse
import lzma
import pickle

def SE(m,Delta):
  x = m*m/Delta
  return x/(x+1)

def logZ(m,Delta,Theta): #VALID ONLY FOR MU=0!!
  x = m*m/Delta
  f1 = 0.5*x + 0.5*np.log(2*3.14159265359/(1+x))
  f2 = 0.5*x - np.log(np.sqrt(2*3.14159265359))-1
  return (1-Theta)*f1 + Theta*f2 - m*m*m/(3*Delta)

def uninf_SE(Tmin,Tmax,NT,Doverrho2, max_iter=10000, tol=1e-6):
    Delta = Doverrho2
    m_init = 1e-6
    m_tab = np.zeros(NT)
    it_tab = np.zeros(NT)
    logZ_tab = np.zeros(NT)
    for i_T,Theta in enumerate(np.linspace(Tmin,Tmax,NT)):
        m_SE = m_init
        for i in range(max_iter):
            m_old = m_SE
            m_SE = 0.5*(Theta + (1-Theta)*SE(m_SE,Delta)) + 0.5 * m_SE
            if (np.abs(m_SE-m_old)/m_SE<tol):
                break
            if  np.abs(m_SE)<1e-50:
                m_SE=0
                break
        m_tab[i_T] = m_SE
        logZ_tab[i_T] = logZ(m_SE,Delta,Theta)
        it_tab[i_T] = i
    return m_tab,it_tab,logZ_tab

def inf_SE(Tmin,Tmax,NT,Doverrho2, max_iter=10000, tol=1e-6):
    Delta = Doverrho2
    m_init = 1
    m_tab = np.zeros(NT)
    it_tab = np.zeros(NT)
    logZ_tab = np.zeros(NT)
    for i_T,Theta in enumerate(np.linspace(Tmin,Tmax,NT)):
        m_SE = m_init
        for i in range(max_iter):
            m_old = m_SE
            m_SE = 0.5*(Theta + (1-Theta)*SE(m_SE,Delta)) + 0.5 * m_SE
            if (np.abs(m_SE-m_old)/m_SE<tol):
                break
            if  np.abs(m_SE)<1e-50:
                m_SE=0
                break
        m_tab[i_T] = m_SE
        logZ_tab[i_T] = logZ(m_SE,Delta,Theta)
        it_tab[i_T] = i
    return m_tab,it_tab,logZ_tab

def main():
    parser = argparse.ArgumentParser(description="Sample")
    #parser.add_argument("--N", type=int, default=10000, help="size")
    parser.add_argument("--Doverrho2", type=float, default=2.0, help="Delta over rho squared")
    parser.add_argument("--Tmin", type=float, default=0.01, help="Tmin")
    parser.add_argument("--Tmax", type=float, default=0.01, help="Tmax")
    parser.add_argument("--NT", type=int, default=10, help="NT")
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
    save_dir = args.save_dir

    m_uninf,it_uninf,logZ_un = uninf_SE(Tmin,Tmax,NT,Doverrho2)
    m_inf,it_inf,logZ_in = inf_SE(Tmin,Tmax,NT,Doverrho2)
    diff = m_inf-m_uninf
    diff_logZ = logZ_in-logZ_un

    keys = [r"$\theta$","step","# iter (uninf)","# iter (inf)", r'$m_{uninf}$', r'$m_{inf}$', r'$\delta m$', r'$\log Z_{uninf}$', r'$\log Z_{inf}$', r'$\delta logZ$', r'$\Delta/\rho^2$', "Tmin", "Tmax", "NT"]
    dict_list=[]
    for i_T,Theta in enumerate(np.linspace(Tmin,Tmax,NT)):
        values = [Theta,i_T,it_uninf[i_T],it_inf[i_T], m_uninf[i_T], m_inf[i_T], diff[i_T], logZ_un[i_T],logZ_in[i_T],diff_logZ[i_T], Doverrho2, Tmin, Tmax, NT]
        dict_list.append(dict(zip(keys, values)))
    df = pd.DataFrame(dict_list)
    #dff = pd.DataFrame(values,keys)

    file_name = "DF_Deci_SphT_SE_DoR2_" + str(Doverrho2) + "_Tm_" + str(Tmin) + "_TM_" + str(Tmax) + "_NT_" + str(NT) + ".xz"
    with lzma.open(save_dir + file_name, "wb") as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()