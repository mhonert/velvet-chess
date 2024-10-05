/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
use core_affinity::CoreId;


#[cfg(target_os = "linux")]
pub fn pin_thread(cores: &[CoreId]) -> anyhow::Result<()> {
    if cores.is_empty() {
        return Ok(());
    }
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        for &core in cores {
            libc::CPU_SET(core.id, &mut cpuset);
        }

        let result = libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &cpuset);
        if result != 0 {
            return Err(anyhow::anyhow!("Failed to set CPU affinity: {}", result));
        }

        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_thread(cores: &[CoreId]) -> anyhow::Result<()> {
    if cores.is_empty() {
        return Ok(());
    }
    if cores.len() > 1 {
        println!("Warning: CPU pinning to multiple cores is not supported on this platform, ignoring");
        return Ok(());
    }
    
    core_affinity::set_for_current(cores[0]);
}
