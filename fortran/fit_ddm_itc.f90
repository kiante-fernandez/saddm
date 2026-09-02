!===============================================================================
! fit_ddm_itc.f90
!
! Copyright (C) 2026 Kiante Fernandez, <kiantefernan@gmail.com>
!
! GNU GPL v3 - see <http://www.gnu.org/licenses/>.
!
! Per-subject pure DDM fit for ITC. NV = 6:
!   x(1) = a       boundary separation
!   x(2) = t0      non-decision time (seconds)
!   x(3) = z       relative starting bias in (0, 1); 0.5 = midline
!   x(4) = v0      drift intercept
!   x(5) = v_val   value coefficient (USD)
!   x(6) = v_time  delay coefficient (days)
!
! Trial drift: delta_i = v0 + v_val * val_diff_i + v_time * time_diff_i
!
! Same SIMPLEX optimization, 8 restarts (Ratcliff convention).
! Same choice / drift / z conventions as fit_ddm_itc_sa.f90 (FC_M4 style).
! Same diffusion sigma = 1 (matches Stan wiener_lpdf).
!
! No across-trial variability (sa = sv = st = 0). cor_pure evaluates the
! wiener FPT density directly via fc at the population-mean drift, skipping
! the integration loops that fit_ddm_itc_sa.f90 needs.
!
! Input format (one trial per line, whitespace- or comma-delimited):
!   choice  rt  val_diff_usd  time_diff_days
!
! Output: one-row CSV
!   a,t0,z,v0,v_val,v_time,neg_log_lik,n_iter,n_trials
!
! Compile:
!   gfortran -O3 -march=native -funroll-loops -fopenmp -o fit_ddm_itc fit_ddm_itc.f90
! Run:
!   ./fit_ddm_itc <input.csv> <output.csv>
!===============================================================================

module constants_mod
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: PI_VAL = 4.0_dp * atan(1.0_dp)
end module constants_mod

!-------------------------------------------------------------------------------
! Module: wiener FPT density and -log L for a single trial (pure DDM)
!-------------------------------------------------------------------------------
module diffusion_mod
  use constants_mod, only: dp, PI_VAL
  implicit none
contains

  !-----------------------------------------------------------------------------
  ! fc: Wabersich / Navarro-Fuss wiener PDF series at a single drift.
  ! Returns fb * g where fb is a CDF-like quantity and g is the (Gaussian over
  ! drift) factor. For our pure-DDM use we evaluate at u = xb so g is constant
  ! across the t and t+dt calls, and (fc_tpdt - fc_t)/dt gives the FPT density
  ! up to a known constant (which only shifts -log L by an additive constant
  ! and so does not affect optimization). Same routine as fit_ddm_itc_sa.f90.
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
  ! cor_pure: -log(likelihood) for one trial of the pure DDM (no sa/sv/st).
  ! zrel is relative starting bias in (0, 1); absolute starting point = a*zrel.
  ! Returns chi = -log(density) (up to an additive constant shared across calls).
  !-----------------------------------------------------------------------------
  subroutine cor_pure(aaa, zrel, drift, sss, terr, r, chi)
    implicit none
    real(dp), intent(in) :: aaa, zrel, drift, sss, terr, r
    real(dp), intent(out) :: chi

    real(dp) :: dt, t, z_loc, fake_sc, fc_t, fc_tpdt, dens
    integer  :: nn_loc, kk_loc

    dt = 0.0001_dp
    fake_sc = 1.0_dp     ! avoids divide-by-zero in fc's Gaussian factor;
                         ! constant across t and t+dt so cancels in derivative
    z_loc = aaa * zrel
    t = r - terr

    if (t <= 0.0001_dp) then
      chi = 30.0_dp       ! large penalty for RT below non-decision time
      return
    end if

    nn_loc = 1
    kk_loc = 0
    fc_t    = fc(drift, PI_VAL, drift, sss, aaa, z_loc, drift, fake_sc, t,    nn_loc, kk_loc)
    fc_tpdt = fc(drift, PI_VAL, drift, sss, aaa, z_loc, drift, fake_sc, t+dt, nn_loc, kk_loc)

    dens = (fc_tpdt - fc_t) / dt
    if (dens > 0.0_dp) then
      chi = -log(dens)
    else
      chi = 30.0_dp
    end if
  end subroutine cor_pure

end module diffusion_mod

!-------------------------------------------------------------------------------
! Module: shared trial data
!-------------------------------------------------------------------------------
module trial_data_mod
  use constants_mod, only: dp
  implicit none
  integer, parameter :: MAX_TRIALS = 2000
  integer :: n_trials
  real(dp) :: rt(MAX_TRIALS)
  integer  :: mch(MAX_TRIALS)
  real(dp) :: val_diff(MAX_TRIALS)
  real(dp) :: time_diff(MAX_TRIALS)
  real(dp) :: min_rt   ! min RT in trial set; sets upper bound on t0 (Stan convention)
end module trial_data_mod

!-------------------------------------------------------------------------------
! Module: objective function
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
    real(dp) :: s_val, a_val, t0_val, z_val, v0_val, v_val_coef, v_time_coef
    real(dp) :: zrel, drift, drift_pop, chi
    integer :: i, j

    s_val = 1.0_dp       ! diffusion sigma = 1 (matches Stan wiener_lpdf)

    ! Clamp to plausible ITC ranges
    if (x(1) < 0.3_dp)  x(1) = 0.3_dp
    if (x(1) > 6.0_dp)  x(1) = 6.0_dp
    a_val = x(1)

    ! t0 clamped below by 0.05 and above by min_rt - 0.01 (matches Stan
    ! wiener_lpdf <upper=min_rt> tau, prevents the t0-pinning pathology).
    if (x(2) < 0.05_dp)              x(2) = 0.05_dp
    if (x(2) > min_rt - 0.01_dp)     x(2) = min_rt - 0.01_dp
    t0_val = x(2)

    if (x(3) < 0.05_dp) x(3) = 0.05_dp
    if (x(3) > 0.95_dp) x(3) = 0.95_dp
    z_val = x(3)

    v0_val      = x(4)
    v_val_coef  = x(5)
    v_time_coef = x(6)

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

      call cor_pure(a_val, zrel, drift, s_val, t0_val, rt(j), chi)
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
! Module: Nelder-Mead simplex (same as fit_ddm_itc_sa.f90)
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
    trace = .false.

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
program fit_ddm_itc
  !$ use omp_lib
  use constants_mod, only: dp
  use trial_data_mod
  use simplex_mod, only: simplx
  implicit none

  integer, parameter :: NV = 6
  integer, parameter :: N_STARTS = 5
  real(dp) :: x(NV), s(NV), best_x(NV)
  real(dp) :: crit_val, y_best, best_nll
  integer  :: itmax, itrace, iopt, iter_out, ios, j, i
  integer  :: start, best_init, best_iter
  character(255) :: in_path, out_path
  integer  :: ich, n_read
  real(dp) :: rt_in, vd_in, td_in
  real(dp) :: rnd6(NV)
  integer, allocatable :: seed_array(:)
  integer :: seed_size

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'Usage: ./fit_ddm_itc <input.csv> <output.csv>'
    write(*, '(A)') '  Trial format: choice  rt  val_diff_usd  time_diff_days'
    stop 1
  end if

  write(*, '(A,A)') 'Reading: ', trim(in_path)
  open(unit=1, file=trim(in_path), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write(*, '(A,A)') 'ERROR: cannot open: ', trim(in_path)
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

  ! Compute min_rt for the t0 upper clamp
  min_rt = rt(1)
  do i = 2, n_trials
    if (rt(i) < min_rt) min_rt = rt(i)
  end do
  write(*, '(A,I0,A,F8.4)') '  trials = ', n_trials, '  min_rt = ', min_rt

  !$ call omp_set_num_threads(4)

  ! ---------------------------------------------------------------------------
  ! Multi-start optimization: 5 random starting values, each running 50 SIMPLEX
  ! restarts. Track the best (lowest neg_log_lik) across all starts.
  ! ---------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed_array(seed_size))
  ! Deterministic seed from output filename hash so re-running gives the same
  ! starts. Falls back to a fixed seed if hashing fails.
  do i = 1, seed_size
    seed_array(i) = 20260518 + i * 17
  end do
  call random_seed(put = seed_array)
  deallocate(seed_array)

  best_nll = huge(1.0_dp)
  best_init = 0

  do start = 1, N_STARTS
    ! Perturb starting values: spatial params (a, t0, z, v0) +/- 20%,
    ! drift coefs (v_val, v_time) +/- 50%. Start 1 uses unperturbed defaults.
    if (start == 1) then
      x(1) = 2.50_dp;  x(2) = min(0.30_dp, min_rt - 0.05_dp)
      x(3) = 0.47_dp;  x(4) = 0.10_dp
      x(5) = 0.05_dp;  x(6) = -0.005_dp
    else
      call random_number(rnd6)
      x(1) = 2.50_dp  * (1.0_dp + 0.4_dp * (rnd6(1) - 0.5_dp))
      x(2) = min(0.30_dp, min_rt - 0.05_dp) * (1.0_dp + 0.4_dp * (rnd6(2) - 0.5_dp))
      x(3) = 0.47_dp  + 0.30_dp * (rnd6(3) - 0.5_dp)
      x(4) = 0.10_dp  + 0.60_dp * (rnd6(4) - 0.5_dp)
      x(5) = 0.05_dp  * (1.0_dp + 1.0_dp * (rnd6(5) - 0.5_dp))
      x(6) = -0.005_dp * (1.0_dp + 1.0_dp * (rnd6(6) - 0.5_dp))
    end if

    crit_val = 1.0E-4_dp
    itrace   = 0
    y_best   = huge(1.0_dp)
    do j = 1, 50
      itmax = 150
      iopt  = 50

      s(1) = max(abs(x(1)) / 20.0_dp, 0.01_dp)   ! a
      s(2) = max(abs(x(2)) / 20.0_dp, 0.005_dp)  ! t0
      s(3) = max(abs(x(3)) / 20.0_dp, 0.01_dp)   ! z
      s(4) = max(abs(x(4)) / 20.0_dp, 0.005_dp)  ! v0
      s(5) = max(abs(x(5)) / 20.0_dp, 1.0e-3_dp) ! v_val
      s(6) = max(abs(x(6)) / 20.0_dp, 3.0e-4_dp) ! v_time

      call simplx(x, s, crit_val, itmax, itrace, iopt, NV, iter_out, y_best)
    end do
    write(*, '(A,I1,A,I3,A,E14.6)') ' start ', start, ' final_iter=', iter_out, &
                                    ' neg_log_lik=', y_best
    write(*, '(A,6F12.5)') '   x=', x(1:6)
    flush(6)

    if (y_best < best_nll) then
      best_nll  = y_best
      best_init = start
      best_x    = x
      best_iter = iter_out
    end if
  end do

  x = best_x
  y_best = best_nll
  iter_out = best_iter

  open(unit=12, file=trim(out_path), status='replace', action='write', iostat=ios)
  if (ios /= 0) then
    write(*, '(A,A)') 'ERROR: cannot open output: ', trim(out_path)
    stop 1
  end if
  write(12, '(A)') 'a,t0,z,v0,v_val,v_time,neg_log_lik,n_iter,n_trials,best_init'
  write(12, '(F12.6,",",F12.6,",",F12.6,",",F12.6,",",ES14.6,",",ES14.6,",",ES14.6,",",I0,",",I0,",",I0)') &
            x(1), x(2), x(3), x(4), x(5), x(6), y_best, iter_out, n_trials, best_init
  close(12)

  write(*, '(/,A,A)') 'Wrote: ', trim(out_path)
  write(*, '(A,I0,A,E14.6)') 'Best start: ', best_init, '  neg_log_lik = ', y_best
  write(*, '(A)') 'a, t0, z, v0, v_val, v_time'
  write(*, '(6F12.5)') x(1:6)
end program fit_ddm_itc
