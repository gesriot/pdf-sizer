//! Raster helpers for MRC segmentation: float planes, sliding min/max, box blur,
//! area resampling, push-pull hole filling, binary dilation and connected components.

use std::collections::VecDeque;

use super::types::ComponentStats;

/// Per-output-pixel source span and coverage weights for box (area) resampling.
fn axis_weights(n: usize, nn: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = n as f64 / nn as f64;
    (0..nn)
        .map(|i| {
            let a = i as f64 * scale;
            let b = (i + 1) as f64 * scale;
            let start = (a.floor() as usize).min(n - 1);
            let end = (b.ceil() as usize).clamp(start + 1, n);
            let weights = (start..end)
                .map(|k| ((b.min((k + 1) as f64) - a.max(k as f64)).max(0.0) / scale) as f32)
                .collect();
            (start, weights)
        })
        .collect()
}

/// Box-filter resampling: every output pixel is the coverage-weighted mean of the source
/// pixels it overlaps. Exact for downscaling, nearest-like for upscaling.
pub(crate) fn area_resize(src: &[f32], w: usize, h: usize, nw: usize, nh: usize) -> Vec<f32> {
    let xw = axis_weights(w, nw);
    let yw = axis_weights(h, nh);
    let mut tmp = vec![0f32; h * nw];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out = &mut tmp[y * nw..(y + 1) * nw];
        for (o, (start, weights)) in out.iter_mut().zip(&xw) {
            *o = weights.iter().enumerate().map(|(k, &wt)| row[start + k] * wt).sum();
        }
    }
    let mut out = vec![0f32; nh * nw];
    for (y, (start, weights)) in yw.iter().enumerate() {
        let o = &mut out[y * nw..(y + 1) * nw];
        for (k, &wt) in weights.iter().enumerate() {
            let r = &tmp[(start + k) * nw..(start + k + 1) * nw];
            for (a, &b) in o.iter_mut().zip(r) {
                *a += b * wt;
            }
        }
    }
    out
}

/// Bilinear resampling with pixel-centre alignment (used to upsample smooth estimates).
pub(crate) fn bilinear_resize(src: &[f32], w: usize, h: usize, nw: usize, nh: usize) -> Vec<f32> {
    let coords = |n: usize, nn: usize| -> Vec<(usize, usize, f32)> {
        (0..nn)
            .map(|i| {
                let p = ((i as f32 + 0.5) * n as f32 / nn as f32 - 0.5).clamp(0.0, (n - 1) as f32);
                let i0 = p.floor() as usize;
                let i1 = (i0 + 1).min(n - 1);
                (i0, i1, p - i0 as f32)
            })
            .collect()
    };
    let xs = coords(w, nw);
    let ys = coords(h, nh);
    let mut out = vec![0f32; nw * nh];
    for (y, &(y0, y1, fy)) in ys.iter().enumerate() {
        let r0 = &src[y0 * w..(y0 + 1) * w];
        let r1 = &src[y1 * w..(y1 + 1) * w];
        for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
            let top = r0[x0] + (r0[x1] - r0[x0]) * fx;
            let bottom = r1[x0] + (r1[x1] - r1[x0]) * fx;
            out[y * nw + x] = top + (bottom - top) * fy;
        }
    }
    out
}

/// Sliding-window maximum (or minimum) over `[i - r, i + r]`, O(n) with a monotonic deque.
fn extreme_1d(line: &[f32], out: &mut [f32], r: usize, max: bool, dq: &mut VecDeque<usize>) {
    let n = line.len();
    let better = |a: f32, b: f32| if max { a >= b } else { a <= b };
    dq.clear();
    let mut next = 0;
    for (i, o) in out.iter_mut().enumerate() {
        let hi = (i + r).min(n - 1);
        while next <= hi {
            while dq.back().is_some_and(|&b| better(line[next], line[b])) {
                dq.pop_back();
            }
            dq.push_back(next);
            next += 1;
        }
        let lo = i.saturating_sub(r);
        while dq.front().is_some_and(|&f| f < lo) {
            dq.pop_front();
        }
        *o = line[dq[0]];
    }
}

/// Separable square max/min filter of radius `r` (grey-level dilation/erosion).
fn extreme_filter(src: &[f32], w: usize, h: usize, r: usize, max: bool) -> Vec<f32> {
    let mut dq = VecDeque::new();
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        extreme_1d(&src[y * w..(y + 1) * w], &mut tmp[y * w..(y + 1) * w], r, max, &mut dq);
    }
    let mut out = vec![0f32; w * h];
    let mut col = vec![0f32; h];
    let mut res = vec![0f32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = tmp[y * w + x];
        }
        extreme_1d(&col, &mut res, r, max, &mut dq);
        for y in 0..h {
            out[y * w + x] = res[y];
        }
    }
    out
}

pub(crate) fn max_filter(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    extreme_filter(src, w, h, r, true)
}

pub(crate) fn min_filter(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    extreme_filter(src, w, h, r, false)
}

fn box_1d(line: &[f32], out: &mut [f32], r: usize) {
    let n = line.len();
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0f64);
    for &v in line {
        prefix.push(prefix.last().unwrap() + v as f64);
    }
    for (i, o) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(r);
        let hi = (i + r + 1).min(n);
        *o = ((prefix[hi] - prefix[lo]) / (hi - lo) as f64) as f32;
    }
}

/// Separable mean filter of radius `r`; the window is clipped (not padded) at the edges.
pub(crate) fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        box_1d(&src[y * w..(y + 1) * w], &mut tmp[y * w..(y + 1) * w], r);
    }
    let mut out = vec![0f32; w * h];
    let mut col = vec![0f32; h];
    let mut res = vec![0f32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = tmp[y * w + x];
        }
        box_1d(&col, &mut res, r);
        for y in 0..h {
            out[y * w + x] = res[y];
        }
    }
    out
}

/// Push-pull hole filling. `values` holds `ch` interleaved channels; `conf` in [0, 1] says
/// how much each pixel's own value is trusted (0 = hole). Holes receive a smooth blend of
/// the surrounding known values, coarse levels filling large holes.
pub(crate) fn push_pull(values: &mut [f32], conf: &[f32], w: usize, h: usize, ch: usize) {
    if w <= 1 && h <= 1 {
        return;
    }
    let total: f32 = conf.iter().sum();
    if total <= 0.0 {
        return;
    }
    let sw = w.div_ceil(2);
    let sh = h.div_ceil(2);
    let mut small = vec![0f32; sw * sh * ch];
    let mut small_conf = vec![0f32; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let c = conf[y * w + x];
            if c <= 0.0 {
                continue;
            }
            let si = (y / 2) * sw + x / 2;
            small_conf[si] += c;
            for k in 0..ch {
                small[si * ch + k] += values[(y * w + x) * ch + k] * c;
            }
        }
    }
    for (i, c) in small_conf.iter_mut().enumerate() {
        if *c > 0.0 {
            for k in 0..ch {
                small[i * ch + k] /= *c;
            }
            *c = c.min(1.0);
        }
    }
    push_pull(&mut small, &small_conf, sw, sh, ch);
    for k in 0..ch {
        let plane: Vec<f32> = (0..sw * sh).map(|i| small[i * ch + k]).collect();
        let up = bilinear_resize(&plane, sw, sh, w, h);
        for (i, &u) in up.iter().enumerate() {
            let c = conf[i].clamp(0.0, 1.0);
            let v = &mut values[i * ch + k];
            *v = *v * c + u * (1.0 - c);
        }
    }
}

/// Square binary dilation of radius `r` (separable, O(n)).
pub(crate) fn dilate(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return mask.to_vec();
    }
    let spread = |line: &[bool], out: &mut [bool]| {
        let n = line.len();
        let mut last: Option<usize> = None;
        // Forward pass: within r after a set pixel.
        for i in 0..n {
            if line[i] {
                last = Some(i);
            }
            out[i] = last.is_some_and(|l| i - l <= r);
        }
        let mut next: Option<usize> = None;
        for i in (0..n).rev() {
            if line[i] {
                next = Some(i);
            }
            if next.is_some_and(|nx| nx - i <= r) {
                out[i] = true;
            }
        }
    };
    let mut tmp = vec![false; w * h];
    for y in 0..h {
        spread(&mask[y * w..(y + 1) * w], &mut tmp[y * w..(y + 1) * w]);
    }
    let mut out = vec![false; w * h];
    let mut col = vec![false; h];
    let mut res = vec![false; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = tmp[y * w + x];
        }
        spread(&col, &mut res);
        for y in 0..h {
            out[y * w + x] = res[y];
        }
    }
    out
}

fn find(parent: &mut [u32], mut a: u32) -> u32 {
    while parent[a as usize] != a {
        parent[a as usize] = parent[parent[a as usize] as usize];
        a = parent[a as usize];
    }
    a
}

/// 8-connected component labelling. Returns per-pixel labels (0 = background, components
/// numbered from 1) and the stats of each component (index = label - 1).
pub(crate) fn label_components(mask: &[bool], w: usize, h: usize) -> (Vec<u32>, Vec<ComponentStats>) {
    let mut labels = vec![0u32; w * h];
    let mut parent: Vec<u32> = vec![0];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !mask[i] {
                continue;
            }
            let mut neighbours = [0u32; 4];
            let mut n = 0;
            if x > 0 && labels[i - 1] != 0 {
                neighbours[n] = labels[i - 1];
                n += 1;
            }
            if y > 0 {
                let up = i - w;
                if x > 0 && labels[up - 1] != 0 {
                    neighbours[n] = labels[up - 1];
                    n += 1;
                }
                if labels[up] != 0 {
                    neighbours[n] = labels[up];
                    n += 1;
                }
                if x + 1 < w && labels[up + 1] != 0 {
                    neighbours[n] = labels[up + 1];
                    n += 1;
                }
            }
            if n == 0 {
                let l = parent.len() as u32;
                parent.push(l);
                labels[i] = l;
            } else {
                let mut root = find(&mut parent, neighbours[0]);
                for &nb in &neighbours[1..n] {
                    let other = find(&mut parent, nb);
                    if other != root {
                        let (lo, hi) = (root.min(other), root.max(other));
                        parent[hi as usize] = lo;
                        root = lo;
                    }
                }
                labels[i] = root;
            }
        }
    }
    // Compact the roots into consecutive labels.
    let mut remap = vec![0u32; parent.len()];
    let mut stats: Vec<ComponentStats> = Vec::new();
    for (i, label) in labels.iter_mut().enumerate() {
        if *label == 0 {
            continue;
        }
        let root = find(&mut parent, *label) as usize;
        if remap[root] == 0 {
            stats.push(ComponentStats::new((i % w) as u32, (i / w) as u32));
            remap[root] = stats.len() as u32;
        }
        *label = remap[root];
        stats[*label as usize - 1].add((i % w) as u32, (i / w) as u32);
    }
    (labels, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_resize_averages_blocks() {
        let src = [0.0, 2.0, 4.0, 6.0, 0.0, 2.0, 4.0, 6.0];
        let out = area_resize(&src, 4, 2, 2, 1);
        assert_eq!(out, vec![1.0, 5.0]);
        let out = area_resize(&src, 4, 2, 3, 1);
        assert!((out[1] - 3.0).abs() < 1e-5, "{out:?}");
    }

    #[test]
    fn min_max_filters_cover_radius() {
        let mut src = vec![5.0f32; 25];
        src[12] = 0.0;
        let mn = min_filter(&src, 5, 5, 1);
        assert_eq!(mn[6], 0.0);
        assert_eq!(mn[0], 5.0);
        // Opening with the same radius keeps the 3×3 dark square and nothing more.
        let mx = max_filter(&mn, 5, 5, 1);
        assert_eq!(mx[12], 0.0);
        assert_eq!(mx[0], 5.0);
    }

    #[test]
    fn push_pull_fills_holes_smoothly() {
        let w = 8;
        let mut vals: Vec<f32> = (0..w * w).map(|i| if i % w < 4 { 10.0 } else { 30.0 }).collect();
        let mut conf = vec![1.0f32; w * w];
        for y in 2..6 {
            for x in 2..6 {
                conf[y * w + x] = 0.0;
                vals[y * w + x] = 0.0;
            }
        }
        push_pull(&mut vals, &conf, w, w, 1);
        for y in 2..6 {
            for x in 2..6 {
                let v = vals[y * w + x];
                assert!((10.0..=30.0).contains(&v), "hole value {v} at {x},{y}");
            }
        }
    }

    #[test]
    fn dilate_and_label() {
        let mut m = vec![false; 49];
        m[24] = true;
        let d = dilate(&m, 7, 7, 1);
        assert_eq!(d.iter().filter(|&&b| b).count(), 9);
        let mut two = vec![false; 49];
        two[0] = true;
        two[8] = true; // diagonal neighbour: same component
        two[48] = true;
        let (labels, stats) = label_components(&two, 7, 7);
        assert_eq!(stats.len(), 2);
        assert_eq!(labels[0], labels[8]);
        assert_eq!(stats[0].area, 2);
    }
}
