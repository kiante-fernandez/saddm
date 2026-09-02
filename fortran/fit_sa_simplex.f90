!===============================================================================
! fit_sa_simplex.f90
!
! Copyright (C) 2024 Blair Shevlin & Kianté Fernandez, <kiantefernan@gmail.com>
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.
!
! Record of Revisions
!
! Date            Programmers                         Descriptions of Change
! ====         ================                       ======================
! 2024/01/01    Blair Shevlin                          wrote original code (fit_sa_SIMPLEX.f)
! 2026/03/13    Kianté Fernandez                       refactored to modern Fortran 90+
!
! Modern Fortran 90+ refactor of fit_sa_SIMPLEX.f
! Diffusion model fitting via Nelder-Mead simplex optimization.
!
! Changes from original:
!   - Free-format source (no column restrictions)
!   - implicit none everywhere
!   - Modules for shared data, RNG, numerical routines
!   - Replaced Intel MKL VSL with Fortran intrinsic random_number
!   - Replaced ASSIGN/computed GOTO with modern control flow
!   - Replaced numbered DO loops with DO...END DO
!   - Replaced numbered CONTINUE with labeled END DO or removed
!   - OpenMP directives preserved (optional — works with or without -fopenmp)
!   - All subroutines/functions use explicit interfaces via modules
!
! Compile (macOS with Homebrew gfortran):
!   gfortran -O2 -fopenmp -o fit_sa_simplex fit_sa_simplex.f90
!
! Without OpenMP:
!   gfortran -O2 -o fit_sa_simplex fit_sa_simplex.f90
!
! Run:
!   ./fit_sa_simplex <datafile_suffix>
!   e.g. ./fit_sa_simplex ".e.fast-dm.csv"
!===============================================================================

module constants_mod
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)  ! double precision kind
  real(dp), parameter :: PI_VAL = 4.0_dp * atan(1.0_dp)
end module constants_mod

!-------------------------------------------------------------------------------
! Module: random number generation (replaces MKL VSL)
!-------------------------------------------------------------------------------
module rng_mod
  use constants_mod, only: dp
  implicit none
contains

  subroutine ranunif(gu, nit, seed)
    ! Generate nit uniform random numbers in [0,1) using Fortran intrinsic RNG
    integer, intent(in) :: nit, seed
    real(dp), intent(out) :: gu(:)
    integer :: i
    integer, allocatable :: seed_array(:)
    integer :: seed_size

    call random_seed(size=seed_size)
    allocate(seed_array(seed_size))
    do i = 1, seed_size
      seed_array(i) = seed + (i - 1) * 37
    end do
    call random_seed(put=seed_array)
    call random_number(gu(1:nit))
    deallocate(seed_array)
  end subroutine ranunif

end module rng_mod

!-------------------------------------------------------------------------------
! Module: numerical integration (Gauss quadrature) and diffusion model PDF
!-------------------------------------------------------------------------------
module diffusion_mod
  use constants_mod, only: dp, PI_VAL
  implicit none
contains

  !-----------------------------------------------------------------------------
  ! FC: core diffusion model density at a single drift rate u
  !-----------------------------------------------------------------------------
  real(dp) function fc(u, pi_in, uu, s, a, z, xb, sc, t, nn_in, ktorp)
    implicit none
    real(dp), intent(in) :: u, pi_in, uu, s, a, z, xb, sc, t
    integer, intent(in) :: nn_in, ktorp

    real(dp) :: xlim, test_val, b_val, c_val, d_val, e_val, g_val, h_val
    real(dp) :: sf, r_val, rr_val, gg_val, ff_val, q_val, x_val, ex_val, y_val, xx_val, fb_val
    integer :: n_idx, m_val

    ! Suppress unused-argument warnings (inherited from original interface)
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

    if (ktorp /= 1) then
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
  end function fc

  !-----------------------------------------------------------------------------
  ! GQ: Gauss quadrature integration
  !-----------------------------------------------------------------------------
  real(dp) function gq(a_lo, b_hi, n_pts, pi_in, u, s, ag, z, xb, sc, tt, nn_in, ktorp)
    implicit none
    real(dp), intent(in) :: a_lo, b_hi, pi_in, u, s, ag, z, xb, sc, tt
    integer, intent(in) :: n_pts, nn_in, ktorp

    ! Gauss-Legendre points and weights (up to 20-point rule)
    real(dp) :: pt(100), wt(100), wz(9), p_loc(20), w_loc(20)
    real(dp) :: ba, sum_val, baa
    integer :: nstart, nd2, j_idx, nj, nstj, locate

    ! Initialize quadrature data
    call init_quadrature_data(pt, wt, wz)

    nstart = (n_pts / 2) * ((n_pts - 1) / 2)
    nd2 = n_pts / 2
    ba = b_hi - a_lo

    do j_idx = 1, nd2
      nj = n_pts - j_idx + 1
      nstj = nstart + j_idx
      p_loc(j_idx) = pt(nstj)
      p_loc(nj) = -p_loc(j_idx)
      w_loc(j_idx) = wt(nstj)
      w_loc(nj) = w_loc(j_idx)
    end do

    if (nd2 * 2 /= n_pts) then
      p_loc(nd2 + 1) = 0.0_dp
      w_loc(nd2 + 1) = wz(nd2)
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
  ! FFC: integrates FC over drift rate variability
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
    zzz = 0.0_dp

    ffc = gq(aa_lo, zzz, nn_in, pi_in, u, s, a, z, xb, sc, tt, nn_in, ktorp) + &
          gq(zzz, bb_hi, nn_in, pi_in, u, s, a, z, xb, sc, tt, nn_in, ktorp)
  end function ffc

  !-----------------------------------------------------------------------------
  ! COR: compute -log(likelihood) for a single trial
  !-----------------------------------------------------------------------------
  subroutine cor(aaa, zz, xxb, sss, terr, acc, r, scc, m, sg, chi, st, &
                 x1, x2, pz, pxa)
    implicit none
    real(dp), intent(in) :: aaa, zz, xxb, sss, terr, r, scc, sg, st, pz, pxa
    real(dp), intent(inout) :: acc, x1, x2
    integer, intent(inout) :: m
    real(dp), intent(out) :: chi

    real(dp) :: dt, sc_loc, s_loc, a_loc, xb_loc, z_loc, ter_loc
    real(dp) :: gw, gww, pzz, t, ts, y_val, xx_val, accc, t3
    integer :: nsz, nnsz, i6, it
    real(dp) :: ti_loc  ! intentionally uninitialized; ffc overwrites via tt = t
    integer :: nn_loc, kk_loc

    ! Suppress unused-argument warnings (inherited from original interface)
    if (.false.) then; acc = 0.0_dp; x1 = 0.0_dp; x2 = 0.0_dp; dt = zz; end if

    dt = 0.0001_dp
    m = 5
    nn_loc = 1
    ter_loc = terr
    sc_loc = scc
    s_loc = sss
    nsz = 15
    nnsz = 1 + nsz / 2
    gw = 1.0_dp / real(nsz, dp)
    gww = gw * gw
    pzz = pz / pxa
    xb_loc = xxb
    chi = 0.0_dp
    ts = r - ter_loc

    ! First pass: compute cumulative distribution for contamination
    a_loc = aaa - real(nnsz, dp) * sg * gw
    kk_loc = 0
    t3 = 2.0_dp
    accc = 0.0_dp
    do i6 = 1, nsz
      xb_loc = xxb
      a_loc = a_loc + sg * gw
      z_loc = a_loc / 2.0_dp
      accc = accc + gw * ffc(t3, PI_VAL, xb_loc, s_loc, a_loc, z_loc, &
                              xb_loc, sc_loc, ti_loc, nn_loc, kk_loc)
    end do

    ! Second pass: compute likelihood
    a_loc = aaa - real(nnsz, dp) * sg * gw
    y_val = 0.0_dp
    do i6 = 1, nsz
      a_loc = a_loc + sg * gw
      z_loc = a_loc / 2.0_dp
      t = ts - real(nnsz, dp) * st * gw
      do it = 1, nsz
        t = t + st * gw
        if (t < 0.0001_dp) then
          xx_val = pzz * accc * gww
        else
          xb_loc = xxb
          xx_val = (ffc(t + dt, PI_VAL, xb_loc, s_loc, a_loc, z_loc, &
                        xb_loc, sc_loc, ti_loc, nn_loc, kk_loc) - &
                    ffc(t, PI_VAL, xb_loc, s_loc, a_loc, z_loc, &
                        xb_loc, sc_loc, ti_loc, nn_loc, kk_loc)) / dt
        end if
        y_val = y_val + xx_val * gww * (1.0_dp - pz) + pzz * accc * gww
      end do
    end do

    if (y_val > 0.0_dp) chi = -log(y_val)
  end subroutine cor

  !-----------------------------------------------------------------------------
  ! Initialize Gauss quadrature points and weights (DATA statements → arrays)
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
! Module: shared trial data (replaces scratch file I/O)
!-------------------------------------------------------------------------------
module trial_data_mod
  use constants_mod, only: dp
  implicit none
  integer, parameter :: MAX_TRIALS = 1000
  integer :: n_trials
  real(dp) :: rt(MAX_TRIALS)
  integer  :: mcond(MAX_TRIALS), mch(MAX_TRIALS)
end module trial_data_mod

!-------------------------------------------------------------------------------
! Module: objective function (FOFS)
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
    real(dp) :: pxa, s_val, pz, a_val, z_val, terr_val, sc_val, sg, st
    real(dp) :: zz, v, vv, rr, chi
    real(dp) :: x1_loc, x2_loc, a1_loc
    integer :: i, j, jj, m_loc

    ! Find max RT for contamination scaling
    pxa = 0.0_dp
    do i = 1, n_trials
      if (rt(i) > pxa) pxa = rt(i)
    end do
    pxa = pxa / 1000.0_dp

    s_val = 0.1_dp

    ! Contamination parameter is fixed at 0.0001 for this model variant.
    ! The bound check is vestigial from the original code.
    if (x(7) < 0.0_dp) x(7) = 0.0_dp
    x(7) = 0.0001_dp
    pz = x(7)

    m_loc = 5

    if (x(1) < 0.065_dp) x(1) = 0.065_dp
    if (x(1) > 0.240_dp) x(1) = 0.240_dp
    a_val = x(1)
    x(5) = a_val / 2.0_dp
    z_val = x(5)

    if (x(2) > 0.640_dp) x(2) = 0.640_dp
    terr_val = x(2)

    if (x(3) < 0.01_dp) x(3) = 0.01_dp
    if (x(3) > 0.3_dp) x(3) = 0.3_dp
    sc_val = x(3)

    if (x(6) > 0.45_dp) x(6) = 0.45_dp
    if (x(6) <= 0.03_dp) x(6) = 0.03_dp
    st = x(6)

    if (x(4) < 0.0002_dp) x(4) = 0.0002_dp
    sg = x(4)

    ! Compute -log(likelihood) for each trial (parallelized)
    !$omp parallel do private(j, jj, zz, v, vv, rr, chi, x1_loc, x2_loc, a1_loc)
    do j = 1, n_trials
      jj = mcond(j)
      vv = x(7 + jj)

      if (mch(j) == 1) then
        zz = z_val
        v = -vv
      else  ! mch(j) == 0
        zz = a_val - z_val
        v = vv
      end if

      rr = rt(j) / 1000.0_dp

      call cor(a_val, zz, v, s_val, terr_val, a1_loc, rr, sc_val, m_loc, sg, chi, st, &
               x1_loc, x2_loc, pz, pxa)
      xml(j) = chi
    end do
    !$omp end parallel do

    ! Sum negative log-likelihoods
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

  subroutine simplx(x, scale, crit, itmax, itrace, iopt, nv)
    implicit none
    integer, intent(in) :: nv
    real(dp), intent(inout) :: x(nv), scale(nv)
    real(dp), intent(in) :: crit
    integer, intent(inout) :: itmax, itrace, iopt

    real(dp) :: pl(40), p(40, 39), y(40), pstar(39), pbar(39)
    integer :: matrix(5, 5)
    real(dp) :: alph, bet, gamma_val, fnv, fnvp1
    real(dp) :: t1, t2
    integer :: nvp1, iter, il, ih, i, j, i2, lastm, it
    logical :: trace, first_trace

    alph = 1.0_dp
    bet = 0.5_dp
    gamma_val = 2.0_dp
    trace = .true.

    if (crit <= 0.0_dp) return  ! invalid criterion
    if (itmax <= 0) itmax = 1

    matrix = 0
    lastm = 1
    iter = 1
    nvp1 = nv + 1
    fnv = real(nv, dp)
    fnvp1 = real(nvp1, dp)
    first_trace = .true.

    ! Generate a regular simplex
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

    ! Move centroid to starting vector and scale
    do j = 1, nv
      do i = 1, nvp1
        p(i, j) = p(i, j) * scale(j) + x(j)
      end do
    end do

    ! Evaluate function at each simplex vertex
    do i = 1, nvp1
      write(*, '(A,I3,2X,39F7.3)') ' VERTEX', i, (p(i, j), j=1, nv)
      do j = 1, nv
        pstar(j) = p(i, j)
      end do
      y(i) = fofs(nv, pstar)
    end do

    ! Find initial minimum
    call find_min(y, nvp1, il)

    if (first_trace) then
      write(*, '(A)') '                         NO    IL   FUNCTION    CRITERION   PARAMETERS'
      first_trace = .false.
    end if

    ! Main iteration loop
    main_loop: do while (iter < itmax)
      ! Find maximum Y (worst point)
      t1 = y(1)
      ih = 1
      trace = .false.
      it = iter - 1
      if (mod(it, itrace) == 0) trace = .true.

      ! Check for no improvement
      if (mod(it, iopt) == 0 .and. it /= 0) then
        if (all(abs(p(il, 1:nv) - pl(1:nv)) <= 0.00000001_dp)) then
          write(*, '(/,A,I3,A)') 'NO IMPROVEMENT IN', iopt, ' TRIALS'
          exit main_loop
        end if
      end if

      if (mod(it, iopt) == 0) then
        pl(1:nv) = p(il, 1:nv)
      end if

      do i = 2, nvp1
        if (y(i) > t1) then
          t1 = y(i)
          ih = i
        end if
      end do

      ! Compute centroid excluding worst point
      do j = 1, nv
        t1 = 0.0_dp
        do i = 1, nvp1
          if (i /= ih) t1 = p(i, j) + t1
        end do
        pbar(j) = t1 / fnv
      end do

      ! Try reflection
      do j = 1, nv
        pstar(j) = (1.0_dp + alph) * pbar(j) - alph * p(ih, j)
      end do
      t1 = fofs(nv, pstar)

      if (t1 <= y(il)) then
        ! Reflection succeeded, try expansion
        do j = 1, nv
          p(ih, j) = pstar(j)
          pstar(j) = (1.0_dp + gamma_val) * pstar(j) - gamma_val * pbar(j)
        end do
        t2 = t1
        t1 = fofs(nv, pstar)
        il = ih

        if (t1 <= t2) then
          ! Expansion succeeded
          if (trace) write(*, '(A)') ' EXPANSION SUCCEEDED'
          matrix(lastm, 1) = matrix(lastm, 1) + 1
          lastm = 1
          y(ih) = t1
          p(ih, 1:nv) = pstar(1:nv)
        else
          ! Reflection succeeded but expansion failed
          if (trace) write(*, '(A)') ' REFLECTION SUCCEEDED'
          matrix(lastm, 2) = matrix(lastm, 2) + 1
          lastm = 2
          y(ih) = t2
          ! p(ih,:) already set above
        end if
      else
        ! Reflection failed — find second-highest Y
        t2 = y(il)
        i2 = il
        do i = 1, nvp1
          if (i /= ih .and. y(i) > t2) then
            t2 = y(i)
            i2 = i
          end if
        end do

        if (t1 < t2) then
          ! Normal move
          if (trace) write(*, '(A)') ' NORMAL MOVE'
          matrix(lastm, 3) = matrix(lastm, 3) + 1
          lastm = 3
          y(ih) = t1
          p(ih, 1:nv) = pstar(1:nv)
        else
          ! Try contraction
          if (t1 < y(ih)) then
            ! Exchange pstar and p(ih,:)
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
            ! Contraction succeeded
            if (trace) write(*, '(A)') ' CONTRACTION SUCCEEDED'
            matrix(lastm, 4) = matrix(lastm, 4) + 1
            lastm = 4
            if (t1 < y(il)) il = ih
            y(ih) = t1
            p(ih, 1:nv) = pstar(1:nv)
          else
            ! Contraction failed — shrink simplex toward best point
            if (trace) write(*, '(A)') ' CONTRACTION FAILED'
            matrix(lastm, 5) = matrix(lastm, 5) + 1
            lastm = 5
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

      ! Check convergence
      t1 = sum(y(1:nvp1)) / fnvp1
      t2 = sqrt(sum((y(1:nvp1) - t1)**2) / fnv)

      if (t2 < crit) exit main_loop

      ! Find current minimum
      call find_min(y, nvp1, il)

      if (trace) then
        write(*, '(1X,20X,2I6,2E12.4,39F7.3)') iter, il, y(il), t2, (p(il, j), j=1, nv)
      end if

      iter = iter + 1
    end do main_loop

    if (iter >= itmax) then
      write(*, '(A)') ' MAXIMUM NUMBER OF ITERATIONS'
    end if

    ! Copy best parameters back to x
    x(1:nv) = p(il, 1:nv)

    ! Print summary
    write(*, '(/,29X,A)') 'SUBSEQUENT MOVE'
    write(*, '(A)') '                         EXP REF NOR CON FLD'
    write(*, '(A,5I5)') '                  EXP', (matrix(1, j), j=1, 5)
    write(*, '(A,5I5)') '                  REF', (matrix(2, j), j=1, 5)
    write(*, '(A,5I5)') ' PREVIOUS         NOR', (matrix(3, j), j=1, 5)
    write(*, '(A,5I5)') ' MOVE             CON', (matrix(4, j), j=1, 5)
    write(*, '(A,5I5)') '                  FLD', (matrix(5, j), j=1, 5)

    do i = 1, nvp1
      write(*, '(A,I3,2X,39F7.3)') ' VERTEX', i, (p(i, j), j=1, nv)
    end do

    write(*, '(/,A,I4,1X,A,I2,1X,A,E11.4,3X,A,E11.4)') &
      'ITER=', iter, 'IL=', il, 'Y(IL)=', y(il), 'CRITE', t2
    write(*, '(A,39F10.3)') ' BEST PARAMETER ESTIMATES', (p(il, j), j=1, nv)
    write(*, '(A)') 'a ter eta sa z st po v1 v2 v3 v4 logL'
    write(*, '(20F10.4)') (p(il, j), j=1, nv), y(il)
    write(12, '(9F10.4,F14.4,10F10.4)') (p(il, j), j=1, nv), y(il)
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
program fit_sa_simplex
  !$ use omp_lib
  use constants_mod, only: dp
  use trial_data_mod
  use simplex_mod, only: simplx
  implicit none

  integer, parameter :: NV = 11
  real(dp) :: x(NV), s(NV)
  real(dp) :: crit_val
  integer :: n, mmc, ios
  integer :: itmax, itrace, iopt
  integer :: i, j, k, ich
  real(dp) :: rr
  character(8) :: aa
  character(255) :: arg1
  character(95) :: d1, d2, d3, d4, d5, ff

  ! Read command line argument (data file suffix)
  call get_command_argument(1, arg1)
  d2 = trim(arg1)

  ! Setup filename prefixes
  d1 = 'subj00'
  d4 = 'subj0'
  d5 = 'subj'
  d1 = adjustl(trim(d1))
  d4 = adjustl(trim(d4))
  d5 = adjustl(trim(d5))
  d2 = adjustl(trim(d2))

  ! Trial count
  n = 1000

  ! Set OpenMP threads (if compiled with -fopenmp)
  mmc = 64
  !$ call omp_set_num_threads(mmc)

  ! Process subjects — adjust loop bounds for multi-subject runs
  do k = 1, 1
    ! Build filename
    write(d3, '(I2)') k
    d3 = adjustl(d3)
    ff = trim(d1) // trim(d3) // trim(d2)
    if (k > 9) ff = trim(d4) // trim(d3) // trim(d2)
    if (k > 99) then
      write(d3, '(I3)') k
      d3 = adjustl(d3)
      ff = trim(d5) // trim(d3) // trim(d2)
    end if

    print *, trim(ff)

    ! Open results output file
    open(unit=12, file='results.txt', status='replace', action='write', iostat=ios)
    if (ios /= 0) stop 'ERROR: Cannot open results.txt for writing'

    ! Read data file
    open(unit=1, file=trim(ff), status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*, '(A,A)') 'ERROR: Cannot open data file: ', trim(ff)
      stop 1
    end if
    do i = 1, n
      read(1, *) ich, rr, aa
      rt(i) = 1000.0_dp * rr
      mch(i) = ich
      ! Map condition names to integers
      if (aa == 'high')    mcond(i) = 1
      if (aa == 'low')     mcond(i) = 2
      if (aa == 'vlow')    mcond(i) = 3
      if (aa == 'nonword') mcond(i) = 4
    end do
    close(1)

    ! Store trial count in module variable
    n_trials = n

    ! Initial parameter guesses
    ! x(1)=boundary, x(2)=ter, x(3)=eta, x(4)=sa, x(5)=z, x(6)=st, x(7)=po
    ! x(8..11) = drift rates for 4 conditions
    x(1) = 0.11_dp    ! boundary
    x(2) = 0.22_dp    ! ter (non-decision time)
    x(3) = 0.23_dp    ! eta (drift variability)
    x(4) = 0.10_dp    ! sa (boundary variability)
    x(5) = x(1) / 2.0_dp  ! z (starting point = a/2)
    x(6) = 0.10_dp    ! st (non-decision time variability)
    x(7) = 0.0001_dp  ! po (contamination)
    x(8) = 0.32_dp    ! v1 (drift rate condition 1)
    x(9) = 0.20_dp    ! v2 (drift rate condition 2)
    x(10) = 0.1_dp    ! v3 (drift rate condition 3)
    x(11) = 0.01_dp   ! v4 (drift rate condition 4)

    ! Run simplex optimizer twice
    do j = 1, 2
      crit_val = 1.0E-10_dp
      itmax = 400
      if (j == 2) itmax = 800
      itrace = 20
      iopt = itmax

      ! Set step sizes
      do i = 1, NV
        s(i) = x(i) / 20.0_dp
      end do

      call simplx(x, s, crit_val, itmax, itrace, iopt, NV)
    end do

    write(*, '(F7.4,9F8.4)') (x(i), i=1, NV)
    close(12)
  end do

  stop
end program fit_sa_simplex
