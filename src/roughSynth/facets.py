import numpy as np
import matplotlib
import matplotlib.colors as colors

def gen_normals(gridX, gridY, fzzs):

    fsx = np.abs(gridX[0, 1] - gridX[0, 0])
    fsy = np.abs(gridY[1, 0] - gridY[0, 0])

    # first compute surface normals via central differences
    dZdx = np.zeros_like(fzzs)
    dZdy = np.zeros_like(fzzs)
    dZdx[:, 1:-1] = (fzzs[:, 2:] - fzzs[:, :-2]) / (2 * fsx)
    dZdy[1:-1, :] = (fzzs[2:, :] - fzzs[:-2, :]) / (2 * fsy)

    # fill in edges with copy
    dZdx[:, 0] = dZdx[:, 1]
    dZdx[:, -1] = dZdx[:, -2]
    dZdy[0, :] = dZdy[1, :]
    dZdy[-1, :] = dZdy[-2, :]

    # compute normal vectors
    norms = np.sqrt(dZdx**2 + dZdy**2 + 1)
    n_hat = np.zeros((gridX.shape[1], gridX.shape[0], 3))
    n_hat[:, :, 0] = -dZdx / norms
    n_hat[:, :, 1] = -dZdy / norms
    n_hat[:, :, 2] = 1 / norms

    # generate other two basis using reference vector (pointing in x direction)
    ref = np.zeros_like(n_hat)
    ref[:, :, 0] = 1

    # if n_hat is nearly parallel with ref, switch to y direction
    parallel_mask = np.abs(n_hat[:, :, 0]) > 0.9
    ref[parallel_mask, :] = [0, 1, 0]

    # get first tangent with cross product
    u_hat = np.cross(ref, n_hat)
    u_hat /= np.linalg.norm(u_hat, axis=2, keepdims=True)

    # get second tangent with cross product
    v_hat = np.cross(n_hat, u_hat)

    return n_hat, u_hat, v_hat


def export_obj_points_colored(filename, xs, ys, zs, values, nxs, nys, nzs, cmap_name="magma", vmin=None, vmax=None, nscale=0.5e3):

    # Normalize values to [0, 1]
    if vmin and vmax:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    print(f"Exporting obj to: {filename}")

    with open(filename, "w") as f:
        f.write("# Point cloud OBJ with vertex colors\n")
        i = 0
        for x, y, z, v, nx, ny, nz in zip(xs, ys, zs, values, nxs, nys, nzs):
            r, g, b, _ = cmap(norm(v))
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            #f.write(f"v {x+nx*nscale:.6f} {y+ny*nscale:.6f} {z+nz*nscale:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            #f.write(f"l {i+1} {i+2}\n")
            i += 1
            if i % 66 == 0:
                print(f"Writing out facet data ... {i}/{len(xs)*2}", end="      \r")



def export_obj_grid_colored(filename, gridX, gridY, fzzs, cmap_name="magma", vmin=None, vmax=None):
    # Normalize values to [0, 1]
    values = fzzs.ravel()
    if vmin is not None and vmax is not None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))

    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    nrows, ncols = fzzs.shape
    xs = gridX.ravel()
    ys = gridY.ravel()
    zs = fzzs.ravel()

    print(f"Exporting obj to: {filename}")

    with open(filename, "w") as f:
        f.write("# Colored grid mesh OBJ\n")

        # Write vertices with colors
        for x, y, z in zip(xs, ys, zs):
            r, g, b, _ = cmap(norm(z))
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

        # Write quad faces using grid connectivity
        # OBJ is 1-based indexing
        for i in range(nrows - 1):
            for j in range(ncols - 1):
                v1 = i * ncols + j + 1
                v2 = i * ncols + (j + 1) + 1
                v3 = (i + 1) * ncols + (j + 1) + 1
                v4 = (i + 1) * ncols + j + 1
                f.write(f"f {v1} {v2} {v3} {v4}\n")

                

def export_facets(fxs, fys, fzs, fnxs, fnys, fnzs, fuxs, fuys, fuzs, fvxs, fvys, fvzs, filename, obj=None):
    
    # export into facet file
    with open(filename, 'w') as f:
        i = 0
        for x, y, z, nx, ny, nz, ux, uy, uz, vx, vy, vz in zip(fxs, fys, fzs, fnxs, fnys, fnzs, fuxs, fuys, fuzs, fvxs, fvys, fvzs):
            if i != len(fxs) - 1:
                f.write(f"{x:.6f},{y:.6f},{z:.6f}:{nx:.6f},{ny:.6f},{nz:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}\n")
            else:
                f.write(f"{x:.6f},{y:.6f},{z:.6f}:{nx:.6f},{ny:.6f},{nz:.6f}:{ux:.6f},{uy:.6f},{uz:.6f}:{vx:.6f},{vy:.6f},{vz:.6f}")
            i += 1
        print(f"Exported facet data to: {filename}")

    if obj:
        export_obj_points_colored(
            obj,
            fxs, fys, fzs,
            fzs,
            fnxs, fnys, fnzs,
            cmap_name="magma", nscale=500
        )


def export_grid_to_facet(gridX, gridY, fzzs, facet_filename, obj_filename=None):
    n_hat, u_hat, v_hat = gen_normals(gridX, gridY, fzzs)

    xs = gridX.ravel()
    ys = gridY.ravel()
    zs = fzzs.ravel()

    nxs = n_hat[:, :, 0].ravel()
    nys = n_hat[:, :, 1].ravel()
    nzs = n_hat[:, :, 2].ravel()

    uxs = u_hat[:, :, 0].ravel()
    uys = u_hat[:, :, 1].ravel()
    uzs = u_hat[:, :, 2].ravel()

    vxs = v_hat[:, :, 0].ravel()
    vys = v_hat[:, :, 1].ravel()
    vzs = v_hat[:, :, 2].ravel()

    export_facets(
        xs, ys, zs,
        nxs, nys, nzs,
        uxs, uys, uzs,
        vxs, vys, vzs,
        facet_filename,
        obj=None   # keep facet export unchanged
    )

    if obj_filename:
        export_obj_grid_colored(obj_filename, gridX, gridY, fzzs, cmap_name="magma")

