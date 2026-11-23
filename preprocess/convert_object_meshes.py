"""Step 3 of the benchmark build: HOT3D object models (.glb) -> .ply meshes.

    --step library   <assets>/ -> data/hot3d/previous/
                     copies the object models and writes instance.json without the headsets,
                     the keyboard and the cellphone (objects that are not in the benchmark)
    --step export    <assets>/<id>.glb -> data/hot3d/obj/<id>.ply
                     full-resolution meshes as written by trimesh (no vertex merging)
    --step simplify  data/hot3d/obj/<id>.ply -> data/hot3d/pre-process/<id>.ply
                     quadric edge-collapse decimation to at most 4,000 vertices with pymeshlab,
                     the recipe of Text2HOI's preprocessing_grab_object.py; read by the data loader

export and simplify reproduce the meshes used in the paper byte for byte (export tested with
trimesh 3.23.5 and 4.11.0, simplify with pymeshlab==2021.10; the filter name below is the 2021.10
one, and other pymeshlab versions may give slightly different meshes). pymeshlab 2021.10 has
wheels only for Python <= 3.9:

    python preprocess/convert_object_meshes.py --step library  --assets /path/to/hot3d/assets
    python preprocess/convert_object_meshes.py --step export   --assets /path/to/hot3d/assets
    pip install pymeshlab==2021.10
    python preprocess/convert_object_meshes.py --step simplify --assets /path/to/hot3d/assets

data/hot3d/object_mesh/ (used for obj.pkl and by the evaluator) is NOT produced by this script;
see README.md, "Building the benchmark from HOT3D", step 3.
"""
import argparse
import glob
import json
import os
import os.path as osp
import shutil

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


# Object-library entries that are not benchmark objects (headsets, and the two objects
# that the preprocessing drops).
NON_BENCHMARK = ("aria_d1283", "aria_g1283", "quest3", "keyboard", "cellphone")


def copy_library(assets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(osp.join(assets, "*"))):
        if osp.isfile(path) and osp.basename(path) != "instance.json":
            shutil.copy(path, out_dir)
    with open(osp.join(assets, "instance.json")) as f:
        instance = json.load(f)
    instance = {k: v for k, v in instance.items() if v["instance_name"] not in NON_BENCHMARK}
    with open(osp.join(out_dir, "instance.json"), "w") as f:
        json.dump(instance, f, indent=4)
    print(f"{len(instance)} objects -> {out_dir}")


def export_glb(assets, out_dir):
    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    for glb_path in sorted(glob.glob(osp.join(assets, "*.glb"))):
        out_path = osp.join(out_dir, osp.basename(glb_path)[:-4] + ".ply")
        trimesh.load(glb_path, force="mesh").export(out_path)
        print(out_path)


def simplify(src_dir, save_folder, n_target_vertices=4000):
    import pymeshlab as ml
    import tqdm

    os.makedirs(save_folder, exist_ok=True)
    original_meshes = sorted(glob.glob(os.path.join(src_dir, "*.ply")))
    for mesh_path in tqdm.tqdm(original_meshes, desc="Processing object meshes"):
        basename = os.path.basename(mesh_path)
        ms = ml.MeshSet(verbose=0)
        ms.load_new_mesh(mesh_path)
        TARGET = n_target_vertices
        numFaces = 100 + 2 * TARGET
        while ms.current_mesh().vertex_number() > TARGET:
            ms.apply_filter("simplification_quadric_edge_collapse_decimation", targetfacenum=numFaces, preservenormal=True)
            numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)
        m = ms.current_mesh()
        print("output mesh has", m.vertex_number(), "vertex and", m.face_number(), "faces")
        ms.save_current_mesh(osp.join(save_folder, basename))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", required=True, choices=["library", "export", "simplify"])
    parser.add_argument("--assets", required=True,
                        help="HOT3D object library folder (assets/) with <id>.glb and instance.json")
    parser.add_argument("--library_dir", default=osp.join(PROJECT_ROOT, "data/hot3d/previous"))
    parser.add_argument("--obj_dir", default=osp.join(PROJECT_ROOT, "data/hot3d/obj"))
    parser.add_argument("--out_dir", default=osp.join(PROJECT_ROOT, "data/hot3d/pre-process"))
    args = parser.parse_args()

    if args.step == "library":
        copy_library(args.assets, args.library_dir)
    elif args.step == "export":
        export_glb(args.assets, args.obj_dir)
    else:
        simplify(args.obj_dir, args.out_dir)
        instance = osp.join(args.assets, "instance.json")
        if osp.exists(instance):
            shutil.copy(instance, osp.join(args.out_dir, "instance.json"))


if __name__ == "__main__":
    main()
