!===============================================================================
! fit_ddm_itc_sa.f90
!
! Copyright (C) 2026 Kianté Fernandez, <kiantefernan@gmail.com>
!
! GNU GPL v3 — see <http://www.gnu.org/licenses/>.
!
! DDM-SA fit for ITC (intertemporal choice). Adapted from fit_sa_simplex.f90.
!
! Population-level parameters (NV = 9). z is free (FC_M4 convention),
! clamped to [0.05, 0.95]. We do NOT pin z so the SA group fit is
! directly comparable to the per-subject Fortran fits (which also estimate z).
!
!   x(1) = a       boundary separation
!   x(2) = t0      non-decision time (seconds)
!   x(3) = sv      drift-intercept variability (across participants, normal SD)
!   x(4) = sa      boundary-separation variability (across participants, uniform half-range)
!   x(5) = st      non-decision-time variability (across participants, uniform half-range)
!   x(6) = v0      drift intercept (population mean)
!   x(7) = v_val   value coefficient (USD)
!   x(8) = v_time  delay coefficient (days)
!   x(9) = z       relative starting bias in (0, 1); 0.5 = midline
!
! Trial drift:  delta_i = v0 + v_val * val_diff_i + v_time * time_diff_i
! Across-participant: a ~ U(a-sa/2,a+sa/2), t0 ~ U(t0-st/2,t0+st/2), v0 ~ N(v0,sv^2)
!
! Choice convention (matches ddm_itc_raw.stan):
!   choice = 1  LL (upper boundary)  -> cor uses (a, z, -drift)
!   choice = 0  SS (lower boundary)  -> cor uses (a, 1-z, +drift)
!
! Input file format (one trial per line, whitespace-delimited):
!   choice  rt  val_diff_usd  time_diff_days
!
! Output: single CSV row written to the second command-line argument with header
!   a,t0,sv,sa,st,v0,v_val,v_time,neg_log_lik,n_iter,n_trials
!
! Compile (with OpenMP):
!   gfortran -O2 -fopenmp -o fit_ddm_itc_sa fit_ddm_itc_sa.f90
!
! Run:
!   ./fit_ddm_itc_sa <input.csv> <output.csv>
!===============================================================================

module constants_mod
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: PI_VAL = 4.0_dp * atan(1.0_dp)

  ! Order of the double midpoint integration over (a, t0) in COR -- i.e. the
  ! grid that resolves the across-participant spreads sa and st. The original
  ! fit_sa_SIMPLEX.f used 15; a later rewrite cut it to 9 for speed. Restored
  ! to 15. Override at run time with the DDM_NSZ environment variable (used
  ! only for sensitivity checks; cost scales as nsz^2).
  integer :: nsz_global = 15
end module constants_mod

!-------------------------------------------------------------------------------
! Module: numerical integration (Gauss quadrature) and diffusion model PDF
!-------------------------------------------------------------------------------
module diffusion_mod
  use constants_mod, only: dp, PI_VAL, nsz_global
  implicit none

  ! Quadrature abscissae/weights, formerly re-assigned on EVERY call to GQ
  ! (i.e. once per trial per likelihood evaluation, ~1e7 times per fit).
  ! They are constants, so initialize once via init_quadrature_tables() before
  ! any OpenMP region; read-only thereafter, hence safe to share across threads.
  real(dp), save :: qpt(100), qwt(100), qwz(9)

contains

  subroutine init_quadrature_tables()
    implicit none
    call init_quadrature_data(qpt, qwt, qwz)
  end subroutine init_quadrature_tables

  !-----------------------------------------------------------------------------
  ! FC: core diffusion model density at a single drift rate u (verbatim from
  ! fit_sa_simplex.f90)
  !-----------------------------------------------------------------------------
  real(dp) function fc(u, pi_in, uu, s, a, z, xb, sc, t, nn_in, ktorp)
    implicit none
    real(dp), intent(in) :: u, pi_in, uu, s, a, z, xb, sc, t
    integer, intent(in) :: nn_in, ktorp

    real(dp) :: xlim, test_val, b_val, c_val, d_val, e_val, g_val, h_val
    real(dp) :: sf, r_val, rr_val, gg_val, ff_val, q_val, x_val, ex_val, y_val, xx_val, fb_val
    integer :: n_idx, m_val

    if (.false.) then; xlim = uu; m_val = nn_in; end if

    xlim = 0.000001_dp
    test_val = 1.0e-19_dp
    b_val = u / s**2
    c_val = (pi_in * s / a)**2 / 2.0_dp
    d_val = pi_in * z / a
    e_val = c_val * 2.0_dp / pi_in
    g_val = b_val * u / 2.0_dp
    h_val = z * b_val
    sf = 0.0_dp

    if (ktorp == 2) then
      ! FIXED (E4): analytic first-passage DENSITY series,
      !   f(t) = q * sum_n n exp(-r_n t) sin(n d),
      ! the exact t-derivative of the CDF series below. Replaces cor's finite
      ! difference of the CDF, whose catastrophic cancellation produced noise
      ! spikes and sign flips for small t (handoff.md F3, probes C/D).
      m_val = 2000
      do n_idx = 1, m_val
        r_val = g_val + c_val * real(n_idx, dp)**2
        rr_val = r_val * t
        gg_val = sin(real(n_idx, dp) * d_val)
        ff_val = real(n_idx, dp) * exp(-rr_val) * gg_val
        sf = sf + ff_val
        if (n_idx > 20 .and. abs(ff_val) < xlim * abs(sf) .and. &
            abs(test_val) < xlim * abs(sf)) exit
        test_val = ff_val
      end do
    else if (ktorp /= 1) then
      m_val = 1000
      do n_idx = 1, m_val
        r_val = g_val + c_val * real(n_idx, dp)**2
        rr_val = r_val * t
        gg_val = sin(real(n_idx, dp) * d_val)
        ff_val = real(n_idx, dp) * exp(-rr_val) * gg_val / r_val
        sf = sf + ff_val
        if (abs(ff_val) < xlim * sf .and. abs(test_val) < xlim * sf) exit
        test_val = ff_val
      end do
    end if

    q_val = e_val * exp(-h_val)
    g_val = exp(-0.5_dp * ((u - xb) / sc)**2) / (sqrt(2.0_dp * pi_in) * sc)
    x_val = exp(-2.0_dp * h_val)
    ex_val = -2.0_dp * a * b_val
    y_val = exp((ex_val / 2.0_dp))**2
    if (abs(y_val - 1.0_dp) < 1.0e-15_dp) y_val = y_val + 1.0e-10_dp
    xx_val = (y_val - x_val) / (y_val - 1.0_dp)

    fb_val = xx_val - sf * q_val
    fc = fb_val * g_val
    if (ktorp == 1) fc = xx_val * g_val
    if (ktorp == 2) fc = sf * q_val * g_val   ! density mode: sf holds f-series
  end function fc

  !-----------------------------------------------------------------------------
  ! GQ: Gauss quadrature integration (verbatim from fit_sa_simplex.f90)
  !-----------------------------------------------------------------------------
  real(dp) function gq(a_lo, b_hi, n_pts, pi_in, u, s, ag, z, xb, sc, tt, nn_in, ktorp)
    implicit none
    real(dp), intent(in) :: a_lo, b_hi, pi_in, u, s, ag, z, xb, sc, tt
    integer, intent(in) :: n_pts, nn_in, ktorp

    real(dp) :: p_loc(20), w_loc(20)
    real(dp) :: ba, sum_val, baa
    integer :: nstart, nd2, j_idx, nj, nstj, locate

    ! qpt/qwt/qwz are module-level constants set once by init_quadrature_tables()

    nstart = (n_pts / 2) * ((n_pts - 1) / 2)
    nd2 = n_pts / 2
    ba = b_hi - a_lo

    do j_idx = 1, nd2
      nj = n_pts - j_idx + 1
      nstj = nstart + j_idx
      p_loc(j_idx) = qpt(nstj)
      p_loc(nj) = -p_loc(j_idx)
      w_loc(j_idx) = qwt(nstj)
      w_loc(nj) = w_loc(j_idx)
    end do

    if (nd2 * 2 /= n_pts) then
      p_loc(nd2 + 1) = 0.0_dp
      w_loc(nd2 + 1) = qwz(nd2)
    end if

    sum_val = 0.0_dp
    do locate = 1, n_pts
      baa = (ba * p_loc(locate) + (b_hi + a_lo)) / 2.0_dp
      sum_val = sum_val + 0.5_dp * w_loc(locate) * ba * &
                fc(baa, pi_in, u, s, ag, z, xb, sc, tt, nn_in, ktorp)
    end do

    gq = sum_val
  end function gq

  !-----------------------------------------------------------------------------
  ! FFC: integrates FC over drift rate variability (verbatim from fit_sa_simplex.f90)
  !-----------------------------------------------------------------------------
  real(dp) function ffc(t, pi_in, u, s, a, z, xb, sc, tt, nn_in, ktorp)
    implicit none
    real(dp), intent(in) :: t, pi_in, u, s, a, z, xb, sc
    real(dp), intent(inout) :: tt
    integer, intent(inout) :: nn_in
    integer, intent(in) :: ktorp

    real(dp) :: aa_lo, bb_hi, zzz

    tt = t
    nn_in = 11
    aa_lo = xb - 4.5_dp * sc
    bb_hi = xb + 4.5_dp * sc
    ! FIXED (E3): split the two Gauss panels at the Gaussian mean xb instead of
    ! at 0, so the nodes always cover the drift distribution. The original
    ! split at 0 loses the Gaussian entirely when sc is small relative to |xb|
    ! (handoff.md F2, probe A: errors up to x4.5 / x1000 for sc <= 0.02).
    zzz = xb

    ffc = gq(aa_lo, zzz, nn_in, pi_in, u, s, a, z, xb, sc, tt, nn_in, ktorp) + &
          gq(zzz, bb_hi, nn_in, pi_in, u, s, a, z, xb, sc, tt, nn_in, ktorp)
  end function ffc

  !-----------------------------------------------------------------------------
  ! COR: -log(likelihood) for one trial, integrating over sa, st, sv.
  !
  ! Following the M4 food-choice model convention (fit_sa_simplex_fc_m4.f90):
  ! the caller passes zrel as a *proportion* in (0, 1) and inside the sa
  ! integration loop we compute z_loc = zrel * a_loc. That makes the relative
  ! bias scale naturally with the random per-participant boundary.
  !-----------------------------------------------------------------------------
  subroutine cor(aaa, zrel, xxb, sss, terr, r, scc, sg, chi, st, pxa)
    implicit none
    real(dp), intent(in) :: aaa, zrel, xxb, sss, terr, r, scc, sg, st, pxa
    real(dp), intent(out) :: chi

    real(dp) :: dt, sc_loc, s_loc, a_loc, xb_loc, z_loc, ter_loc
    real(dp) :: gw, gww, pz, pzz, t, ts, y_val, xx_val, accc, t3
    integer :: nsz, nnsz, i6, it
    real(dp) :: ti_loc
    integer :: nn_loc, kk_loc

    if (.false.) then; dt = pxa; end if

    dt = 0.0001_dp
    nn_loc = 1
    ter_loc = terr
    sc_loc = scc
    s_loc = sss
    nsz = nsz_global   ! 15 = original fit_sa_SIMPLEX.f order (see constants_mod)
    nnsz = 1 + nsz / 2
    gw = 1.0_dp / real(nsz, dp)
    gww = gw * gw
    pz = 0.0001_dp           ! tiny contamination floor (matches fit_sa_simplex)
    pzz = pz / pxa
    chi = 0.0_dp
    ts = r - ter_loc

    ! First pass: cumulative distribution at t = 2 s for contamination scaling
    a_loc = aaa - real(nnsz, dp) * sg * gw
    kk_loc = 0
    t3 = 2.0_dp
    accc = 0.0_dp
    do i6 = 1, nsz
      xb_loc = xxb
      a_loc = a_loc + sg * gw
      z_loc = zrel * a_loc
      accc = accc + gw * ffc(t3, PI_VAL, xb_loc, s_loc, a_loc, z_loc, &
                              xb_loc, sc_loc, ti_loc, nn_loc, kk_loc)
    end do

    ! Second pass: likelihood with double integration over (sa, st)
    a_loc = aaa - real(nnsz, dp) * sg * gw
    y_val = 0.0_dp
    do i6 = 1, nsz
      a_loc = a_loc + sg * gw
      z_loc = zrel * a_loc
      t = ts - real(nnsz, dp) * st * gw
      do it = 1, nsz
        t = t + st * gw
        if (t < 0.0001_dp) then
          xx_val = pzz * accc * gww
        else
          xb_loc = xxb
          ! FIXED (E4): analytic density (fc ktorp=2) replaces the finite
          ! difference of the CDF (see fc for why).
          xx_val = ffc(t, PI_VAL, xb_loc, s_loc, a_loc, z_loc, &
                       xb_loc, sc_loc, ti_loc, nn_loc, 2)
        end if
        y_val = y_val + xx_val * gww * (1.0_dp - pz) + pzz * accc * gww
      end do
    end do

    ! FIXED (E4): floor at the contamination-only mass so an underflowed or
    ! negative numeric density can never return chi = 0 (likelihood 1).
    if (accc > 0.0_dp) then
      y_val = max(y_val, pzz * accc)
    else
      y_val = max(y_val, 1.0e-12_dp)
    end if
    chi = -log(y_val)
  end subroutine cor

  !-----------------------------------------------------------------------------
  ! Gauss quadrature data (verbatim from fit_sa_simplex.f90)
  !-----------------------------------------------------------------------------
  subroutine init_quadrature_data(pt, wt, wz)
    implicit none
    real(dp), intent(out) :: pt(100), wt(100), wz(9)

    pt( 1) = -5.773502691896259E-01_dp;  pt( 2) = -7.745966692414834E-01_dp
    pt( 3) = -8.611363115940526E-01_dp;  pt( 4) = -3.399810435848563E-01_dp
    pt( 5) = -9.061798459386640E-01_dp;  pt( 6) = -5.384693101056832E-01_dp
    pt( 7) = -9.324695142031520E-01_dp;  pt( 8) = -6.612093864662646E-01_dp
    pt( 9) = -2.386191860831969E-01_dp;  pt(10) = -9.491079123427586E-01_dp
    pt(11) = -7.415311855993942E-01_dp;  pt(12) = -4.058451513773972E-01_dp
    pt(13) = -9.602898564975362E-01_dp;  pt(14) = -7.966664774136267E-01_dp
    pt(15) = -5.255324099163290E-01_dp;  pt(16) = -1.834346424956498E-01_dp
    pt(17) = -9.681602395076261E-01_dp;  pt(18) = -8.360311073266637E-01_dp
    pt(19) = -6.133714327005905E-01_dp;  pt(20) = -3.242534234038089E-01_dp
    pt(21) = -9.739065285171717E-01_dp;  pt(22) = -8.650633666889845E-01_dp
    pt(23) = -6.794095682990245E-01_dp;  pt(24) = -4.333953941292472E-01_dp
    pt(25) = -1.488743389816312E-01_dp;  pt(26) = -9.782286581460570E-01_dp
    pt(27) = -8.870625997680954E-01_dp;  pt(28) = -7.301520055740493E-01_dp
    pt(29) = -5.190961292068119E-01_dp;  pt(30) = -2.695431559523450E-01_dp
    pt(31) = -9.815606342467190E-01_dp;  pt(32) = -9.041172563704749E-01_dp
    pt(33) = -7.699026741943046E-01_dp;  pt(34) = -5.873179542866175E-01_dp
    pt(35) = -3.678314989981802E-01_dp;  pt(36) = -1.252334085114689E-01_dp
    pt(37) = -9.841830547185882E-01_dp;  pt(38) = -9.175983992229781E-01_dp
    pt(39) = -8.015780907333099E-01_dp;  pt(40) = -6.423493394403403E-01_dp
    pt(41) = -4.484927510364468E-01_dp;  pt(42) = -2.304583159551348E-01_dp
    pt(43) = -9.862838086968123E-01_dp;  pt(44) = -9.284348836635734E-01_dp
    pt(45) = -8.272013150697650E-01_dp;  pt(46) = -6.872929048116856E-01_dp
    pt(47) = -5.152486363581541E-01_dp;  pt(48) = -3.191123689278898E-01_dp
    pt(49) = -1.080549487073437E-01_dp;  pt(50) = -9.879925180204854E-01_dp
    pt(51) = -9.372733924007058E-01_dp;  pt(52) = -8.482065834104270E-01_dp
    pt(53) = -7.244177313601699E-01_dp;  pt(54) = -5.709721726085388E-01_dp
    pt(55) = -3.941513470775634E-01_dp;  pt(56) = -2.011940939974345E-01_dp
    pt(57) = -9.894009349916499E-01_dp;  pt(58) = -9.445750230732326E-01_dp
    pt(59) = -8.656312023878317E-01_dp;  pt(60) = -7.554044083550030E-01_dp
    pt(61) = -6.178762444026438E-01_dp;  pt(62) = -4.580167776572274E-01_dp
    pt(63) = -2.816035507792589E-01_dp;  pt(64) = -9.501250983763744E-02_dp
    pt(65) = -9.905754753144173E-01_dp;  pt(66) = -9.506755217687678E-01_dp
    pt(67) = -8.802391537269859E-01_dp;  pt(68) = -7.815140038968014E-01_dp
    pt(69) = -6.576711592166909E-01_dp;  pt(70) = -5.126905370864771E-01_dp
    pt(71) = -3.512317634538763E-01_dp;  pt(72) = -1.784841814958478E-01_dp
    pt(73) = -9.915651684209309E-01_dp;  pt(74) = -9.558239495713978E-01_dp
    pt(75) = -8.926024664975557E-01_dp;  pt(76) = -8.037049589725230E-01_dp
    pt(77) = -6.916870430603533E-01_dp;  pt(78) = -5.597708310739476E-01_dp
    pt(79) = -4.117511614628426E-01_dp;  pt(80) = -2.518862256915055E-01_dp
    pt(81) = -8.477501304173527E-02_dp;  pt(82) = -9.924068438435845E-01_dp
    pt(83) = -9.602081521348301E-01_dp;  pt(84) = -9.031559036148179E-01_dp
    pt(85) = -8.227146565371427E-01_dp;  pt(86) = -7.209661773352294E-01_dp
    pt(87) = -6.005453046616811E-01_dp;  pt(88) = -4.645707413759609E-01_dp
    pt(89) = -3.165640999636298E-01_dp;  pt(90) = -1.603586456402254E-01_dp
    pt(91) = -9.931285991850949E-01_dp;  pt(92) = -9.639719272779138E-01_dp
    pt(93) = -9.122344282513259E-01_dp;  pt(94) = -8.391169718222189E-01_dp
    pt(95) = -7.463319064601507E-01_dp;  pt(96) = -6.360536807265151E-01_dp
    pt(97) = -5.108670019508271E-01_dp;  pt(98) = -3.737060887154196E-01_dp
    pt(99) = -2.277858511416451E-01_dp;  pt(100)= -7.652652113349732E-02_dp

    wt( 1) = 1.000000000000000E+00_dp;   wt( 2) = 5.555555555555557E-01_dp
    wt( 3) = 3.478548451374538E-01_dp;   wt( 4) = 6.521451548625462E-01_dp
    wt( 5) = 2.369268850561891E-01_dp;   wt( 6) = 4.786286704993665E-01_dp
    wt( 7) = 1.713244923791703E-01_dp;   wt( 8) = 3.607615730481386E-01_dp
    wt( 9) = 4.679139345726910E-01_dp;   wt(10) = 1.294849661688697E-01_dp
    wt(11) = 2.797053914892767E-01_dp;   wt(12) = 3.818300505051189E-01_dp
    wt(13) = 1.012285362903762E-01_dp;   wt(14) = 2.223810344533745E-01_dp
    wt(15) = 3.137066458778873E-01_dp;   wt(16) = 3.626837833783620E-01_dp
    wt(17) = 8.127438836157441E-02_dp;   wt(18) = 1.806481606948574E-01_dp
    wt(19) = 2.606106964029355E-01_dp;   wt(20) = 3.123470770400028E-01_dp
    wt(21) = 6.667134430868812E-02_dp;   wt(22) = 1.494513491505806E-01_dp
    wt(23) = 2.190863625159820E-01_dp;   wt(24) = 2.692667193099964E-01_dp
    wt(25) = 2.955242247147529E-01_dp;   wt(26) = 5.566856711617367E-02_dp
    wt(27) = 1.255803694649046E-01_dp;   wt(28) = 1.862902109277342E-01_dp
    wt(29) = 2.331937645919905E-01_dp;   wt(30) = 2.628045445102467E-01_dp
    wt(31) = 4.717533638651183E-02_dp;   wt(32) = 1.069393259953184E-01_dp
    wt(33) = 1.600783285433462E-01_dp;   wt(34) = 2.031674267230659E-01_dp
    wt(35) = 2.334925365383548E-01_dp;   wt(36) = 2.491470458134028E-01_dp
    wt(37) = 4.048400476531588E-02_dp;   wt(38) = 9.212149983772845E-02_dp
    wt(39) = 1.388735102197872E-01_dp;   wt(40) = 1.781459807619457E-01_dp
    wt(41) = 2.078160475368885E-01_dp;   wt(42) = 2.262831802628972E-01_dp
    wt(43) = 3.511946033175186E-02_dp;   wt(44) = 8.015808715976020E-02_dp
    wt(45) = 1.215185706879032E-01_dp;   wt(46) = 1.572031671581935E-01_dp
    wt(47) = 1.855383974779378E-01_dp;   wt(48) = 2.051984637212956E-01_dp
    wt(49) = 2.152638534631578E-01_dp;   wt(50) = 3.075324199611727E-02_dp
    wt(51) = 7.036604748810809E-02_dp;   wt(52) = 1.071592204671719E-01_dp
    wt(53) = 1.395706779261543E-01_dp;   wt(54) = 1.662692058169939E-01_dp
    wt(55) = 1.861610000155622E-01_dp;   wt(56) = 1.984314853271116E-01_dp
    wt(57) = 2.715245941175409E-02_dp;   wt(58) = 6.225352393864789E-02_dp
    wt(59) = 9.515851168249277E-02_dp;   wt(60) = 1.246289712555339E-01_dp
    wt(61) = 1.495959888165767E-01_dp;   wt(62) = 1.691565193950025E-01_dp
    wt(63) = 1.826034150449236E-01_dp;   wt(64) = 1.894506104550685E-01_dp
    wt(65) = 2.414830286854793E-02_dp;   wt(66) = 5.545952937398720E-02_dp
    wt(67) = 8.503614831717917E-02_dp;   wt(68) = 1.118838471934040E-01_dp
    wt(69) = 1.351363684685255E-01_dp;   wt(70) = 1.540457610768103E-01_dp
    wt(71) = 1.680041021564500E-01_dp;   wt(72) = 1.765627053669926E-01_dp
    wt(73) = 2.161601352648331E-02_dp;   wt(74) = 4.971454889496981E-02_dp
    wt(75) = 7.642573025488905E-02_dp;   wt(76) = 1.009420441062872E-01_dp
    wt(77) = 1.225552067114785E-01_dp;   wt(78) = 1.406429146706506E-01_dp
    wt(79) = 1.546846751262652E-01_dp;   wt(80) = 1.642764837458327E-01_dp
    wt(81) = 1.691423829631436E-01_dp;   wt(82) = 1.946178822972648E-02_dp
    wt(83) = 4.481422676569960E-02_dp;   wt(84) = 6.904454273764122E-02_dp
    wt(85) = 9.149002162244999E-02_dp;   wt(86) = 1.115666455473340E-01_dp
    wt(87) = 1.287539625393362E-01_dp;   wt(88) = 1.426067021736066E-01_dp
    wt(89) = 1.527660420658597E-01_dp;   wt(90) = 1.589688433939543E-01_dp
    wt(91) = 1.761400713915212E-02_dp;   wt(92) = 4.060142980038694E-02_dp
    wt(93) = 6.267204833410905E-02_dp;   wt(94) = 8.327674157670474E-02_dp
    wt(95) = 1.019301198172404E-01_dp;   wt(96) = 1.181945319615184E-01_dp
    wt(97) = 1.316886384491766E-01_dp;   wt(98) = 1.420961093183820E-01_dp
    wt(99) = 1.491729864726037E-01_dp;   wt(100)= 1.527533871307258E-01_dp

    wz(1) = 8.888888888888890E-01_dp;  wz(2) = 5.688888888888889E-01_dp
    wz(3) = 4.179591836734694E-01_dp;  wz(4) = 3.302393550012598E-01_dp
    wz(5) = 2.729250867779006E-01_dp;  wz(6) = 2.325515532308739E-01_dp
    wz(7) = 2.025782419255613E-01_dp;  wz(8) = 1.794464703562065E-01_dp
    wz(9) = 1.610544498487837E-01_dp
  end subroutine init_quadrature_data

end module diffusion_mod

!-------------------------------------------------------------------------------
! Module: shared trial data
!-------------------------------------------------------------------------------
module trial_data_mod
  use constants_mod, only: dp
  implicit none
  integer, parameter :: MAX_TRIALS = 35000  ! k=50 x 603 subj = 30150 trials/perm
  integer :: n_trials
  real(dp) :: rt(MAX_TRIALS)        ! response time in seconds
  integer  :: mch(MAX_TRIALS)       ! choice: 1 = LL (upper), 0 = SS (lower)
  real(dp) :: val_diff(MAX_TRIALS)  ! ll_value - ss_value in USD
  real(dp) :: time_diff(MAX_TRIALS) ! ll_time - ss_time in days
end module trial_data_mod

!-------------------------------------------------------------------------------
! Module: objective function (negative log-likelihood)
!-------------------------------------------------------------------------------
module objective_mod
  use constants_mod, only: dp
  use trial_data_mod
  use diffusion_mod
  implicit none
contains

  real(dp) function fofs(nv, x)
    !$ use omp_lib
    implicit none
    integer, intent(in) :: nv
    real(dp), intent(inout) :: x(nv)

    real(dp) :: xml(MAX_TRIALS)
    real(dp) :: pxa, s_val, a_val, t0_val, sv_val, sa_val, z_val, st_val
    real(dp) :: v0_val, v_val_coef, v_time_coef
    real(dp) :: zrel, drift, drift_pop, chi
    integer :: i, j

    ! Find max RT for contamination scaling
    pxa = 0.0_dp
    do i = 1, n_trials
      if (rt(i) > pxa) pxa = rt(i)
    end do
    if (pxa <= 0.0_dp) pxa = 1.0_dp

    s_val = 1.0_dp     ! diffusion sigma: matches Stan's wiener_lpdf (sigma=1)

    ! Clamp to plausible ITC ranges (raw USD/days)
    if (x(1) < 0.3_dp)  x(1) = 0.3_dp     ! a >= 0.3
    if (x(1) > 6.0_dp)  x(1) = 6.0_dp     ! a <= 6
    a_val = x(1)

    if (x(2) < 0.05_dp) x(2) = 0.05_dp    ! t0 >= 0.05 s
    if (x(2) > 1.0_dp)  x(2) = 1.0_dp     ! t0 <= 1.0 s
    t0_val = x(2)

    if (x(3) < 0.001_dp) x(3) = 0.001_dp  ! sv (drift sd) >= 0.001 (loose floor)
    if (x(3) > 2.0_dp)   x(3) = 2.0_dp
    sv_val = x(3)

    if (x(4) < 0.0_dp)   x(4) = 0.0_dp    ! sa >= 0
    if (x(4) > a_val)    x(4) = a_val     ! cap sa < a so (a-sa/2)>0
    sa_val = x(4)

    ! st data-dependent upper clamp: 2 * t0 keeps (t0 - st/2, t0 + st/2) > 0
    ! and lets the model express a wide non-decision-time distribution.
    if (x(5) < 0.0_dp)            x(5) = 0.0_dp
    if (x(5) > 2.0_dp * t0_val)   x(5) = 2.0_dp * t0_val
    st_val = x(5)

    ! No bounds on v0/v_val/v_time — drift can be any real
    v0_val      = x(6)
    v_val_coef  = x(7)
    v_time_coef = x(8)

    ! Free relative starting bias z, FC_M4-style clamp [0.05, 0.95].
    if (x(9) < 0.05_dp) x(9) = 0.05_dp
    if (x(9) > 0.95_dp) x(9) = 0.95_dp
    z_val = x(9)

    ! Per-trial -log L. Choice flip mirrors fit_sa_simplex_fc_m4 / lexical:
    !   choice=1 (LL/upper)  ->  zrel = z,    drift negated
    !   choice=0 (SS/lower)  ->  zrel = 1-z,  drift positive
    ! zrel is the proportional starting bias; cor will compute z_loc = a_loc * zrel.
    !$omp parallel do private(j, drift_pop, drift, zrel, chi)
    do j = 1, n_trials
      drift_pop = v0_val + v_val_coef * val_diff(j) + v_time_coef * time_diff(j)

      if (mch(j) == 1) then
        zrel  = z_val
        drift = -drift_pop
      else
        zrel  = 1.0_dp - z_val
        drift = drift_pop
      end if

      call cor(a_val, zrel, drift, s_val, t0_val, rt(j), sv_val, sa_val, &
               chi, st_val, pxa)
      xml(j) = chi
    end do
    !$omp end parallel do

    fofs = 0.0_dp
    do i = 1, n_trials
      fofs = fofs + xml(i)
    end do
  end function fofs

end module objective_mod

!-------------------------------------------------------------------------------
! Module: Nelder-Mead simplex optimizer
!-------------------------------------------------------------------------------
module simplex_mod
  use constants_mod, only: dp
  use objective_mod, only: fofs
  implicit none
contains

  subroutine simplx(x, scale, crit, itmax, itrace, iopt, nv, iter_out, y_best)
    implicit none
    integer, intent(in) :: nv
    real(dp), intent(inout) :: x(nv), scale(nv)
    real(dp), intent(in) :: crit
    integer, intent(inout) :: itmax, itrace, iopt
    integer, intent(out)   :: iter_out
    real(dp), intent(out)  :: y_best

    real(dp) :: pl(40), p(40, 39), y(40), pstar(39), pbar(39)
    real(dp) :: alph, bet, gamma_val, fnv, fnvp1
    real(dp) :: t1, t2
    integer :: nvp1, iter, il, ih, i, j, i2, it
    logical :: trace

    alph = 1.0_dp
    bet = 0.5_dp
    gamma_val = 2.0_dp
    trace = .false.   ! quieter than the lexical fitter

    if (crit <= 0.0_dp) then
      iter_out = 0
      y_best = huge(1.0_dp)
      return
    end if
    if (itmax <= 0) itmax = 1

    iter = 1
    nvp1 = nv + 1
    fnv = real(nv, dp)
    fnvp1 = real(nvp1, dp)

    ! Generate regular simplex
    t1 = (1.0_dp - sqrt(fnvp1)) / sqrt(fnv**3)
    t2 = sqrt(fnvp1 / fnv) + t1

    do i = 1, nv
      do j = 1, nv
        if (i == j) then
          p(i, j) = t2
        else
          p(i, j) = t1
        end if
      end do
    end do

    t1 = -1.0_dp / sqrt(fnv)
    do j = 1, nv
      p(nvp1, j) = t1
    end do

    do j = 1, nv
      do i = 1, nvp1
        p(i, j) = p(i, j) * scale(j) + x(j)
      end do
    end do

    do i = 1, nvp1
      do j = 1, nv
        pstar(j) = p(i, j)
      end do
      y(i) = fofs(nv, pstar)
    end do

    call find_min(y, nvp1, il)

    main_loop: do while (iter < itmax)
      t1 = y(1)
      ih = 1
      it = iter - 1
      if (itrace > 0) trace = mod(it, itrace) == 0

      if (mod(it, iopt) == 0 .and. it /= 0) then
        if (all(abs(p(il, 1:nv) - pl(1:nv)) <= 0.00000001_dp)) exit main_loop
      end if

      if (mod(it, iopt) == 0) pl(1:nv) = p(il, 1:nv)

      do i = 2, nvp1
        if (y(i) > t1) then
          t1 = y(i)
          ih = i
        end if
      end do

      do j = 1, nv
        t1 = 0.0_dp
        do i = 1, nvp1
          if (i /= ih) t1 = p(i, j) + t1
        end do
        pbar(j) = t1 / fnv
      end do

      do j = 1, nv
        pstar(j) = (1.0_dp + alph) * pbar(j) - alph * p(ih, j)
      end do
      t1 = fofs(nv, pstar)

      if (t1 <= y(il)) then
        do j = 1, nv
          p(ih, j) = pstar(j)
          pstar(j) = (1.0_dp + gamma_val) * pstar(j) - gamma_val * pbar(j)
        end do
        t2 = t1
        t1 = fofs(nv, pstar)
        il = ih

        if (t1 <= t2) then
          y(ih) = t1
          p(ih, 1:nv) = pstar(1:nv)
        else
          y(ih) = t2
        end if
      else
        t2 = y(il)
        i2 = il
        do i = 1, nvp1
          if (i /= ih .and. y(i) > t2) then
            t2 = y(i)
            i2 = i
          end if
        end do

        if (t1 < t2) then
          y(ih) = t1
          p(ih, 1:nv) = pstar(1:nv)
        else
          if (t1 < y(ih)) then
            do j = 1, nv
              t2 = pstar(j)
              pstar(j) = p(ih, j)
              p(ih, j) = t2
            end do
          end if

          do j = 1, nv
            pstar(j) = (1.0_dp - bet) * pbar(j) + bet * p(ih, j)
          end do
          t1 = fofs(nv, pstar)

          if (t1 < y(ih)) then
            if (t1 < y(il)) il = ih
            y(ih) = t1
            p(ih, 1:nv) = pstar(1:nv)
          else
            do i = 1, nvp1
              if (i /= il) then
                do j = 1, nv
                  p(i, j) = (p(il, j) + p(i, j)) / 2.0_dp
                  pstar(j) = p(i, j)
                end do
                y(i) = fofs(nv, pstar)
              end if
            end do
            call find_min(y, nvp1, il)
          end if
        end if
      end if

      t1 = sum(y(1:nvp1)) / fnvp1
      t2 = sqrt(sum((y(1:nvp1) - t1)**2) / fnv)
      if (t2 < crit) exit main_loop

      call find_min(y, nvp1, il)

      if (trace) then
        write(*, '(1X,A,I5,2X,E12.4,2X,A,9F10.4)') 'iter=', iter, y(il), 'x=', (p(il, j), j=1, nv)
        flush(6)   ! force stdout flush so we can see progress in SGE logs
      end if

      iter = iter + 1
    end do main_loop

    x(1:nv) = p(il, 1:nv)
    iter_out = iter
    y_best = y(il)
  end subroutine simplx

  subroutine find_min(y, n, il)
    implicit none
    real(dp), intent(in) :: y(:)
    integer, intent(in) :: n
    integer, intent(out) :: il
    integer :: i

    il = 1
    do i = 2, n
      if (y(i) < y(il)) il = i
    end do
  end subroutine find_min

end module simplex_mod

!===============================================================================
! Main program
!===============================================================================
program fit_ddm_itc_sa
  !$ use omp_lib
  use constants_mod, only: dp, nsz_global
  use diffusion_mod, only: init_quadrature_tables
  use trial_data_mod
  use simplex_mod, only: simplx
  implicit none

  integer, parameter :: NV = 9
  ! Number of multi-start restarts. Default 5 (needed at k=1 where the
  ! single-trial likelihood is bumpy). Overridable via DDM_NSTARTS for the
  ! high-k runs (k>=10), where the likelihood is smooth and unimodal so fewer
  ! starts converge to the same optimum -- keeps those fits inside the 24h wall.
  integer :: N_STARTS = 5
  real(dp) :: x(NV), s(NV)
  real(dp) :: crit_val, y_best
  integer  :: itmax, itrace, iopt, iter_out, ios, j, i
  character(255) :: in_path, out_path
  integer  :: ich, n_read
  real(dp) :: rt_in, vd_in, td_in
  real(dp) :: x_best(NV), y_best_overall
  real(dp) :: best_x_overall(NV), best_nll_overall, rnd9(NV)
  integer  :: start, best_init
  integer, allocatable :: seed_array(:)
  integer :: seed_size
  character(32) :: env_buf
  integer :: env_len, env_stat, env_nsz, env_nstarts

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'Usage: ./fit_ddm_itc_sa <input.csv> <output.csv>'
    write(*, '(A)') '  Input format (whitespace- or comma-delimited):'
    write(*, '(A)') '    choice rt val_diff_usd time_diff_days   (one row per trial)'
    stop 1
  end if

  write(*, '(A,A)') 'Reading: ', trim(in_path)
  open(unit=1, file=trim(in_path), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write(*, '(A,A)') 'ERROR: Cannot open input file: ', trim(in_path)
    stop 1
  end if

  n_read = 0
  do
    read(1, *, iostat=ios) ich, rt_in, vd_in, td_in
    if (ios /= 0) exit
    n_read = n_read + 1
    if (n_read > MAX_TRIALS) then
      write(*, '(A,I0)') 'ERROR: more than MAX_TRIALS=', MAX_TRIALS
      stop 1
    end if
    mch(n_read)       = ich
    rt(n_read)        = rt_in
    val_diff(n_read)  = vd_in
    time_diff(n_read) = td_in
  end do
  close(1)
  n_trials = n_read
  write(*, '(A,I0)') '  trials = ', n_trials

  ! Thread count is governed by OMP_NUM_THREADS (previously hardcoded to 4,
  ! which silently ignored the environment and any -pe slot allocation).
  ! Results are independent of thread count: the parallel loop writes xml(j)
  ! per trial and the sum over trials is performed serially afterwards.

  ! Quadrature order: default 15, overridable via DDM_NSZ for sensitivity runs.
  call get_environment_variable('DDM_NSZ', env_buf, env_len, env_stat)
  if (env_stat == 0 .and. env_len > 0) then
    read(env_buf(1:env_len), *, iostat=ios) env_nsz
    if (ios == 0 .and. env_nsz >= 3 .and. env_nsz <= 51) nsz_global = env_nsz
  end if
  write(*, '(A,I0)') '  quadrature nsz = ', nsz_global

  ! Multi-start count: default 5, overridable via DDM_NSTARTS for high-k fits.
  call get_environment_variable('DDM_NSTARTS', env_buf, env_len, env_stat)
  if (env_stat == 0 .and. env_len > 0) then
    read(env_buf(1:env_len), *, iostat=ios) env_nstarts
    if (ios == 0 .and. env_nstarts >= 1 .and. env_nstarts <= 20) N_STARTS = env_nstarts
  end if
  write(*, '(A,I0)') '  multistart     = ', N_STARTS
  !$ write(*, '(A,I0)') '  omp threads    = ', omp_get_max_threads()

  ! Fill the quadrature tables once, before any parallel region.
  call init_quadrature_tables()

  ! ---------------------------------------------------------------------------
  ! Multi-start optimization: 5 random starting values, each running 50 SIMPLEX
  ! restarts with best-so-far tracking. Final reported fit = best across all
  ! starts. Helps escape local minima at N=1 where the likelihood is bumpy.
  ! ---------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed_array(seed_size))
  do i = 1, seed_size
    seed_array(i) = 20260518 + i * 23
  end do
  call random_seed(put = seed_array)
  deallocate(seed_array)

  best_nll_overall = huge(1.0_dp)
  best_init        = 0
  best_x_overall   = 0.0_dp

  do start = 1, N_STARTS
    ! Starting values: start 1 uses unperturbed defaults anchored to Stan
    ! random-regime N=117 medians; starts 2-5 perturb spatial params +/-20%
    ! and SA + drift coefs more aggressively to explore the basin landscape.
    if (start == 1) then
      x(1) = 2.80_dp;  x(2) = 0.25_dp;  x(3) = 0.30_dp
      x(4) = 0.40_dp;  x(5) = 0.04_dp;  x(6) = 0.33_dp
      x(7) = 0.0053_dp; x(8) = -0.0016_dp; x(9) = 0.47_dp
    else
      call random_number(rnd9)
      x(1) = 2.80_dp * (1.0_dp + 0.4_dp * (rnd9(1) - 0.5_dp))   ! a +/-20%
      x(2) = 0.25_dp * (1.0_dp + 0.4_dp * (rnd9(2) - 0.5_dp))   ! t0 +/-20%
      x(3) = 0.30_dp * (1.0_dp + 1.0_dp * (rnd9(3) - 0.5_dp))   ! sv +/-50%
      x(4) = 0.40_dp * (1.0_dp + 1.0_dp * (rnd9(4) - 0.5_dp))   ! sa +/-50%
      x(5) = 0.04_dp * (1.0_dp + 1.0_dp * (rnd9(5) - 0.5_dp))   ! st +/-50%
      x(6) = 0.33_dp * (1.0_dp + 0.4_dp * (rnd9(6) - 0.5_dp))   ! v0 +/-20%
      x(7) = 0.0053_dp * (1.0_dp + 1.0_dp * (rnd9(7) - 0.5_dp)) ! v_val +/-50%
      x(8) = -0.0016_dp * (1.0_dp + 1.0_dp * (rnd9(8) - 0.5_dp))! v_time +/-50%
      x(9) = 0.47_dp + 0.30_dp * (rnd9(9) - 0.5_dp)             ! z +/-15
    end if

    crit_val       = 1.0E-4_dp
    itrace         = 0
    y_best_overall = huge(1.0_dp)
    x_best         = x

    do j = 1, 50
      itmax = 150
      iopt  = 50

      s(1) = max(abs(x(1)) / 20.0_dp, 0.01_dp)   ! a
      s(2) = max(abs(x(2)) / 20.0_dp, 0.005_dp)  ! t0
      s(3) = max(abs(x(3)) / 20.0_dp, 0.005_dp)  ! sv
      s(4) = max(abs(x(4)) / 20.0_dp, 0.01_dp)   ! sa
      s(5) = max(abs(x(5)) / 20.0_dp, 0.002_dp)  ! st
      s(6) = max(abs(x(6)) / 20.0_dp, 0.005_dp)  ! v0
      s(7) = max(abs(x(7)) / 20.0_dp, 2.0e-4_dp) ! v_val
      s(8) = max(abs(x(8)) / 20.0_dp, 1.0e-4_dp) ! v_time
      s(9) = max(abs(x(9)) / 20.0_dp, 0.01_dp)   ! z

      call simplx(x, s, crit_val, itmax, itrace, iopt, NV, iter_out, y_best)
      if (y_best < y_best_overall) then
        y_best_overall = y_best
        x_best = x
      end if
    end do

    write(*, '(A,I1,A,E14.6)') ' start ', start, ' best_nll=', y_best_overall
    flush(6)

    if (y_best_overall < best_nll_overall) then
      best_nll_overall = y_best_overall
      best_x_overall   = x_best
      best_init        = start
    end if
  end do

  x      = best_x_overall
  y_best = best_nll_overall

  ! Write one-row CSV with header
  open(unit=12, file=trim(out_path), status='replace', action='write', iostat=ios)
  if (ios /= 0) then
    write(*, '(A,A)') 'ERROR: Cannot open output file: ', trim(out_path)
    stop 1
  end if
  write(12, '(A)') 'a,t0,sv,sa,st,v0,v_val,v_time,z,neg_log_lik,n_iter,n_trials,best_init'
  write(12, '(F12.6,",",F12.6,",",F12.6,",",F12.6,",",F12.6,",",F12.6,",",' // &
            'ES14.6,",",ES14.6,",",F12.6,",",ES14.6,",",I0,",",I0,",",I0)') &
            x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), &
            y_best, iter_out, n_trials, best_init
  close(12)

  write(*, '(/,A,A)') 'Wrote: ', trim(out_path)
  write(*, '(A,I0,A,E14.6)') 'Best start: ', best_init, '  neg_log_lik = ', y_best
  write(*, '(A)') 'a, t0, sv, sa, st, v0, v_val, v_time, z'
  write(*, '(9F10.4)') x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9)

end program fit_ddm_itc_sa
