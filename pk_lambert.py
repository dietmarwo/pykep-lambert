"""Exact Numba port of pykep 2.6 lambert_problem.cpp (Izzo 2015) + calibrated
Keplerian propagation, matching utils_users.compute_transfer bit-for-bit
(including fold-edge needles). All angles/elements per ESA convention:
6th orbital element = MEAN anomaly; mu = 4.90487e12.
"""
import numpy as np
from numba import njit
import math

PI = math.pi

@njit(cache=True)
def _hyperF(z, tol):
    Sj=1.0; Cj=1.0; err=1.0; j=0
    while err>tol:
        Cj1=Cj*(3.0+j)*(1.0+j)/(2.5+j)*z/(j+1)
        Sj1=Sj+Cj1; err=abs(Cj1); Sj=Sj1; Cj=Cj1; j+=1
    return Sj

@njit(cache=True)
def _x2tof2(x, N, lam):
    a=1.0/(1.0-x*x)
    if a>0.0:
        alfa=2.0*math.acos(x)
        beta=2.0*math.asin(math.sqrt(lam*lam/a))
        if lam<0.0: beta=-beta
        return (a*math.sqrt(a)*((alfa-math.sin(alfa))-(beta-math.sin(beta))+2.0*PI*N))/2.0
    else:
        alfa=2.0*math.acosh(x)
        beta=2.0*math.asinh(math.sqrt(-lam*lam/a))
        if lam<0.0: beta=-beta
        return -a*math.sqrt(-a)*((beta-math.sinh(beta))-(alfa-math.sinh(alfa)))/2.0

@njit(cache=True)
def _x2tof(x, N, lam):
    battin=0.01; lagrange=0.2; dist=abs(x-1.0)
    if dist<lagrange and dist>battin:
        return _x2tof2(x,N,lam)
    K=lam*lam; E=x*x-1.0; rho=abs(E); z=math.sqrt(1.0+K*E)
    if dist<battin:
        eta=z-lam*x
        S1=0.5*(1.0-lam-x*eta)
        Q=_hyperF(S1,1e-11); Q=4.0/3.0*Q
        return (eta*eta*eta*Q+4.0*lam*eta)/2.0 + N*PI/(rho**1.5)
    else:
        y=math.sqrt(rho); g=x*z-lam*E
        if E<0.0:
            l=math.acos(g); d=N*PI+l
        else:
            f=y*(z-lam*x); d=math.log(f+g)
        return (x-lam*z-d/y)/E

@njit(cache=True)
def _dTdx(x, T, lam):
    l2=lam*lam; l3=l2*lam; umx2=1.0-x*x
    y=math.sqrt(1.0-l2*umx2); y2=y*y; y3=y2*y
    DT=1.0/umx2*(3.0*T*x-2.0+2.0*l3*x/y)
    DDT=1.0/umx2*(3.0*T+5.0*x*DT+2.0*(1.0-l2)*l3/y3)
    DDDT=1.0/umx2*(7.0*x*DDT+8.0*DT-6.0*(1.0-l2)*l2*l3*x/y3/y2)
    return DT,DDT,DDDT

@njit(cache=True)
def _householder(T, x0, N, eps, iter_max, lam):
    it=0; err=1.0
    while err>eps and it<iter_max:
        tof=_x2tof(x0,N,lam)
        DT,DDT,DDDT=_dTdx(x0,tof,lam)
        delta=tof-T; DT2=DT*DT
        xnew=x0-delta*(DT2-delta*DDT/2.0)/(DT*(DT2-delta*DDT)+DDDT*delta*delta/6.0)
        err=abs(x0-xnew); x0=xnew; it+=1
    return x0

@njit(cache=True)
def _lambert_dv_one(r1, v_dep, r2, v_arr, tof, mu, cw, multi_revs):
    """min dv over all 2Nmax+1 solutions for one cw sense. inf if no solution."""
    c=math.sqrt((r2[0]-r1[0])**2+(r2[1]-r1[1])**2+(r2[2]-r1[2])**2)
    R1=math.sqrt(r1[0]**2+r1[1]**2+r1[2]**2)
    R2=math.sqrt(r2[0]**2+r2[1]**2+r2[2]**2)
    s=(c+R1+R2)/2.0
    ir1=np.empty(3); ir2=np.empty(3)
    for k in range(3):
        ir1[k]=r1[k]/R1; ir2[k]=r2[k]/R2
    ih=np.empty(3)
    ih[0]=ir1[1]*ir2[2]-ir1[2]*ir2[1]
    ih[1]=ir1[2]*ir2[0]-ir1[0]*ir2[2]
    ih[2]=ir1[0]*ir2[1]-ir1[1]*ir2[0]
    ihn=math.sqrt(ih[0]**2+ih[1]**2+ih[2]**2)
    if ihn<1e-300: return np.inf
    for k in range(3): ih[k]/=ihn
    lam2=1.0-c/s
    if lam2<0.0: lam2=0.0
    lam=math.sqrt(lam2)
    it1=np.empty(3); it2=np.empty(3)
    if ih[2]<0.0:
        lam=-lam
        it1[0]=ir1[1]*ih[2]-ir1[2]*ih[1]; it1[1]=ir1[2]*ih[0]-ir1[0]*ih[2]; it1[2]=ir1[0]*ih[1]-ir1[1]*ih[0]
        it2[0]=ir2[1]*ih[2]-ir2[2]*ih[1]; it2[1]=ir2[2]*ih[0]-ir2[0]*ih[2]; it2[2]=ir2[0]*ih[1]-ir2[1]*ih[0]
    else:
        it1[0]=ih[1]*ir1[2]-ih[2]*ir1[1]; it1[1]=ih[2]*ir1[0]-ih[0]*ir1[2]; it1[2]=ih[0]*ir1[1]-ih[1]*ir1[0]
        it2[0]=ih[1]*ir2[2]-ih[2]*ir2[1]; it2[1]=ih[2]*ir2[0]-ih[0]*ir2[2]; it2[2]=ih[0]*ir2[1]-ih[1]*ir2[0]
    n1=math.sqrt(it1[0]**2+it1[1]**2+it1[2]**2); n2=math.sqrt(it2[0]**2+it2[1]**2+it2[2]**2)
    for k in range(3): it1[k]/=n1; it2[k]/=n2
    if cw:
        lam=-lam
        for k in range(3): it1[k]=-it1[k]; it2[k]=-it2[k]
    lam2=lam*lam; lam3=lam*lam2
    T=math.sqrt(2.0*mu/(s*s*s))*tof
    Nmax=int(T/PI)
    T00=math.acos(lam)+lam*math.sqrt(1.0-lam2)
    T0=T00+Nmax*PI
    T1=2.0/3.0*(1.0-lam3)
    if Nmax>0:
        if T<T0:
            it=0; err=1.0; Tmin=T0; x_old=0.0; x_new=0.0
            while True:
                DT,DDT,DDDT=_dTdx(x_old,Tmin,lam)
                if DT!=0.0:
                    x_new=x_old-DT*DDT/(DDT*DDT-DT*DDDT/2.0)
                err=abs(x_old-x_new)
                if err<1e-13 or it>12: break
                Tmin=_x2tof(x_new,Nmax,lam)
                x_old=x_new; it+=1
            if Tmin>T: Nmax-=1
    if multi_revs<Nmax: Nmax=multi_revs
    gamma=math.sqrt(mu*s/2.0); rho=(R1-R2)/c; sigma=math.sqrt(1.0-rho*rho)
    best=np.inf
    nsol=2*Nmax+1
    for idx in range(nsol):
        if idx==0:
            if T>=T00: x0=-(T-T00)/(T-T00+4.0)
            elif T<=T1: x0=T1*(T1-T)/(2.0/5.0*(1.0-lam2*lam3)*T)+1.0
            else: x0=(T/T00)**(0.69314718055994529/math.log(T1/T00))-1.0
            x=_householder(T,x0,0,1e-5,15,lam)
        else:
            i=(idx+1)//2
            if idx%2==1:
                tmp=((i*PI+PI)/(8.0*T))**(2.0/3.0); x0=(tmp-1.0)/(tmp+1.0)
            else:
                tmp=((8.0*T)/(i*PI))**(2.0/3.0); x0=(tmp-1.0)/(tmp+1.0)
            x=_householder(T,x0,i,1e-8,15,lam)
        y=math.sqrt(1.0-lam2+lam2*x*x)
        vr1=gamma*((lam*y-x)-rho*(lam*y+x))/R1
        vr2=-gamma*((lam*y-x)+rho*(lam*y+x))/R2
        vt=gamma*sigma*(y+lam*x); vt1=vt/R1; vt2=vt/R2
        d1=0.0; d2=0.0
        for k in range(3):
            a1=vr1*ir1[k]+vt1*it1[k]-v_dep[k]
            a2=vr2*ir2[k]+vt2*it2[k]-v_arr[k]
            d1+=a1*a1; d2+=a2*a2
        dv=math.sqrt(d1)+math.sqrt(d2)
        if dv<best: best=dv
    return best

@njit(cache=True)
def lambert_dv(r1,v_dep,r2,v_arr,tof,mu,multi_revs):
    a=_lambert_dv_one(r1,v_dep,r2,v_arr,tof,mu,0,multi_revs)
    b=_lambert_dv_one(r1,v_dep,r2,v_arr,tof,mu,1,multi_revs)
    return a if a<b else b

from numba import prange

@njit(cache=True, parallel=True)
def lambert_dv_batch(r1, v_dep, r2, v_arr, tof, mu, multi_revs, out):
    """out[k] = best two-impulse Lambert dv for one already-propagated case.

    This keeps pykep's Lambert problem isolated from Kepler propagation for
    pure solver benchmarks.
    """
    n = tof.shape[0]
    for k in prange(n):
        out[k] = lambert_dv(
            r1[k],
            v_dep[k],
            r2[k],
            v_arr[k],
            tof[k],
            mu,
            int(multi_revs[k]),
        )

# ---- calibrated Keplerian propagation (matches pykep planet.eph) ----
@njit(cache=True)
def _kep_E(M,e):
    E=M+e*math.sin(M)
    for _ in range(8):
        f=E-e*math.sin(E)-M; fp=1.0-e*math.cos(E); E=E-f/fp
    return E

@njit(cache=True)
def state_at_t(a,e,inc,Om,om,M0,n,mu, t_sec, out_r, out_v):
    M=M0+n*t_sec
    E=_kep_E(M,e)
    cE=math.cos(E); sE=math.sin(E)
    nu=2.0*math.atan2(math.sqrt(1.0+e)*math.sin(E/2.0), math.sqrt(1.0-e)*math.cos(E/2.0))
    p=a*(1.0-e*e); r_mag=a*(1.0-e*cE)
    cnu=math.cos(nu); snu=math.sin(nu); sq=math.sqrt(mu/p)
    rpf0=r_mag*cnu; rpf1=r_mag*snu
    vpf0=-sq*snu; vpf1=sq*(e+cnu)
    cO=math.cos(Om); sO=math.sin(Om); ci=math.cos(inc); si=math.sin(inc); co=math.cos(om); so=math.sin(om)
    R00=cO*co-sO*so*ci; R01=-cO*so-sO*co*ci
    R10=sO*co+cO*so*ci; R11=-sO*so+cO*co*ci
    R20=so*si;          R21=co*si
    out_r[0]=R00*rpf0+R01*rpf1; out_r[1]=R10*rpf0+R11*rpf1; out_r[2]=R20*rpf0+R21*rpf1
    out_v[0]=R00*vpf0+R01*vpf1; out_v[1]=R10*vpf0+R11*vpf1; out_v[2]=R20*vpf0+R21*vpf1

@njit(cache=True, parallel=True)
def grid_dv(ai,ei,ii,Oi,oi,M0i,ni, aj,ej,ij,Oj,oj,M0j,nj, mu, tdep, tofs, multi_revs, out):
    """out[A,B] = dv for departure tdep[a] (sec), tof tofs[b] (sec). Parallel over A."""
    A=tdep.shape[0]; B=tofs.shape[0]
    for a in prange(A):
        r1=np.empty(3); v1=np.empty(3); r2=np.empty(3); v2=np.empty(3)
        state_at_t(ai,ei,ii,Oi,oi,M0i,ni,mu, tdep[a], r1, v1)
        for b in range(B):
            state_at_t(aj,ej,ij,Oj,oj,M0j,nj,mu, tdep[a]+tofs[b], r2, v2)
            out[a,b]=lambert_dv(r1,v1,r2,v2,tofs[b],mu,multi_revs)

@njit(cache=True, nogil=True)
def dv_full(ai,ei,ii,Oi,oi,M0i,ni, aj,ej,ij,Oj,oj,M0j,nj, mu, td_sec, tof_sec, multi_revs):
    r1=np.empty(3); v1=np.empty(3); r2=np.empty(3); v2=np.empty(3)
    state_at_t(ai,ei,ii,Oi,oi,M0i,ni,mu, td_sec, r1, v1)
    state_at_t(aj,ej,ij,Oj,oj,M0j,nj,mu, td_sec+tof_sec, r2, v2)
    return lambert_dv(r1,v1,r2,v2,tof_sec,mu,multi_revs)

@njit(cache=True, parallel=True)
def dv_full_batch(A,E,I,O,Oc,M0,NN, mu, i_idx, j_idx, td_sec, tof_sec, max_revs_pair, out):
    """out[k] = dv_full(i[k], j[k], td_sec[k], tof_sec[k]). Parallel over k."""
    n=i_idx.shape[0]
    for k in prange(n):
        i=int(i_idx[k]); j=int(j_idx[k])
        r1=np.empty(3); v1=np.empty(3); r2=np.empty(3); v2=np.empty(3)
        state_at_t(A[i],E[i],I[i],O[i],Oc[i],M0[i],NN[i],mu, td_sec[k], r1, v1)
        state_at_t(A[j],E[j],I[j],O[j],Oc[j],M0[j],NN[j],mu,
                   td_sec[k]+tof_sec[k], r2, v2)
        out[k]=lambert_dv(r1,v1,r2,v2,tof_sec[k],mu,int(max_revs_pair[i,j]))

@njit(cache=True, parallel=True)
def fixed_leg_objective_batch(A,E,I,O,Oc,M0,NN, mu, i, j, current_time_sec,
                              xs, min_tof_sec, max_wait_sec, max_tof_sec,
                              dv_limit, penalty_linear, penalty_quad,
                              multi_revs, values, infos):
    """Evaluate a CMA population for one transfer leg.

    ``xs[:, 0]`` is wait in days and ``xs[:, 1]`` is TOF in days. ``values`` is
    the penalized objective. ``infos`` columns are wait[d], tof[d],
    departure[d], arrival[d], dv[m/s], excess[m/s].
    """
    n=xs.shape[0]
    day=86400.0
    for k in prange(n):
        wait_days=float(xs[k,0])
        tof_days=float(xs[k,1])
        wait_sec=wait_days*day
        tof_sec=tof_days*day
        dep_sec=current_time_sec+wait_sec
        arr_sec=dep_sec+tof_sec

        infos[k,0]=wait_days
        infos[k,1]=tof_days
        infos[k,2]=dep_sec/day
        infos[k,3]=arr_sec/day
        infos[k,4]=np.inf
        infos[k,5]=np.inf
        values[k]=1.0e30

        if wait_sec < -1.0e-12 or wait_sec > max_wait_sec + 1.0e-12:
            continue
        if tof_sec < min_tof_sec - 1.0e-12 or tof_sec > max_tof_sec + 1.0e-12:
            continue

        r1=np.empty(3); v1=np.empty(3); r2=np.empty(3); v2=np.empty(3)
        state_at_t(A[i],E[i],I[i],O[i],Oc[i],M0[i],NN[i],mu, dep_sec, r1, v1)
        state_at_t(A[j],E[j],I[j],O[j],Oc[j],M0[j],NN[j],mu, dep_sec+tof_sec, r2, v2)
        dv=lambert_dv(r1,v1,r2,v2,tof_sec,mu,int(multi_revs))
        if not math.isfinite(dv):
            continue
        excess=dv-dv_limit
        if excess < 0.0:
            excess=0.0
        infos[k,4]=dv
        infos[k,5]=excess
        values[k]=arr_sec/day + penalty_linear*excess + penalty_quad*excess*excess

@njit(cache=True, nogil=True)
def fixed_leg_objective_one(A,E,I,O,Oc,M0,NN, mu, i, j, current_time_sec,
                            wait_days, tof_days, min_tof_sec, max_wait_sec,
                            max_tof_sec, dv_limit, penalty_linear,
                            penalty_quad, multi_revs):
    """Scalar fixed-leg objective for optimizers with a ``fun(x)`` API."""
    day=86400.0
    wait_sec=wait_days*day
    tof_sec=tof_days*day
    dep_sec=current_time_sec+wait_sec
    arr_sec=dep_sec+tof_sec
    dep_day=dep_sec/day
    arr_day=arr_sec/day

    if wait_sec < -1.0e-12 or wait_sec > max_wait_sec + 1.0e-12:
        return 1.0e30, wait_days, tof_days, dep_day, arr_day, np.inf, np.inf
    if tof_sec < min_tof_sec - 1.0e-12 or tof_sec > max_tof_sec + 1.0e-12:
        return 1.0e30, wait_days, tof_days, dep_day, arr_day, np.inf, np.inf

    r1=np.empty(3); v1=np.empty(3); r2=np.empty(3); v2=np.empty(3)
    state_at_t(A[i],E[i],I[i],O[i],Oc[i],M0[i],NN[i],mu, dep_sec, r1, v1)
    state_at_t(A[j],E[j],I[j],O[j],Oc[j],M0[j],NN[j],mu, dep_sec+tof_sec, r2, v2)
    dv=lambert_dv(r1,v1,r2,v2,tof_sec,mu,int(multi_revs))
    if not math.isfinite(dv):
        return 1.0e30, wait_days, tof_days, dep_day, arr_day, dv, np.inf
    excess=dv-dv_limit
    if excess < 0.0:
        excess=0.0
    value=arr_day + penalty_linear*excess + penalty_quad*excess*excess
    return value, wait_days, tof_days, dep_day, arr_day, dv, excess

@njit(cache=True)
def grid_dv_serial(ai,ei,ii,Oi,oi,M0i,ni, aj,ej,ij,Oj,oj,M0j,nj, mu, tdep, tofs, multi_revs, out):
    """Single-threaded grid dv (no prange). For use INSIDE multiprocess workers
    to avoid Numba-thread x process oversubscription."""
    A=tdep.shape[0]; B=tofs.shape[0]
    r1=np.empty(3); v1=np.empty(3); r2=np.empty(3); v2=np.empty(3)
    for a in range(A):
        state_at_t(ai,ei,ii,Oi,oi,M0i,ni,mu, tdep[a], r1, v1)
        for b in range(B):
            state_at_t(aj,ej,ij,Oj,oj,M0j,nj,mu, tdep[a]+tofs[b], r2, v2)
            out[a,b]=lambert_dv(r1,v1,r2,v2,tofs[b],mu,multi_revs)
