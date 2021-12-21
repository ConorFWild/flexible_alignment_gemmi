
  template <typename Func>
  void use_points_in_box_non_overlapping(const Fractional& fctr_, int du, int dv, int dw,
                         Func&& func, bool fail_on_too_large_radius=true) {
    if (fail_on_too_large_radius) {
      if (2 * du >= nu || 2 * dv >= nv || 2 * dw >= nw)
        fail("grid operation failed: radius bigger than half the unit cell?");
    } else {
      // If we'd use the minimum image convention the max would be (nu-1)/2.
      // The limits set here are necessary for index_n() that is used below.
      du = std::min(du, nu - 1);
      dv = std::min(dv, nv - 1);
      dw = std::min(dw, nw - 1);
    }
    const Fractional fctr = fctr_.wrap_to_unit();
    int u0 = iround(fctr.x * nu);
    int v0 = iround(fctr.y * nv);
    int w0 = iround(fctr.z * nw);
    for (int w = w0-dw; w <= w0+dw; ++w)
      for (int v = v0-dv; v <= v0+dv; ++v)
        for (int u = u0-du; u <= u0+du; ++u) {
          //Check if u/v/w and u0/v0/w0 are in different unit cells: they are in
          if (u <0 || u >= nu || v <0 || v >= nv ||w <0 || w >= nw) {
            continue;
          }
          else {

            Fractional fdelta{fctr.x - u * (1.0 / nu),
                              fctr.y - v * (1.0 / nv),
                              fctr.z - w * (1.0 / nw)};
            Position delta = unit_cell.orthogonalize_difference(fdelta);
            func(data[index_n(u, v, w)], delta);
          }
        }
  }

  template <typename Func>
  void use_points_around_non_overlapping(const Fractional& fctr_, double radius, Func&& func,
                         bool fail_on_too_large_radius=true) {
    int du = (int) std::ceil(radius / spacing[0]);
    int dv = (int) std::ceil(radius / spacing[1]);
    int dw = (int) std::ceil(radius / spacing[2]);
    use_points_in_box_non_overlapping(fctr_, du, dv, dw,
                      [&](T& point, const Position& delta) {
                        double d2 = delta.length_sq();
                        if (d2 < radius * radius)
                          func(point, d2);
                      },
                      fail_on_too_large_radius);
  }

  void set_points_around_non_overlapping(const Position& ctr, double radius, T value) {
    Fractional fctr = unit_cell.fractionalize(ctr);
    use_points_around_non_overlapping(fctr, radius, [&](T& point, double) { point = value; });
  }