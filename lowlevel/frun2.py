import subprocess

python_exe = "/opt/miniconda3/envs/IKsim_mj/bin/python"
# script = "lowlevel/build_general_nonh_reverse_lookup.py"
script = "build_general_nonh_reverse_lookup.py"

for rank in range(1, 9):
    if rank in [None]:
        continue
    source = f"a{rank}"
    # target_moves = f"{source}_to_g1 {source}_to_g8"
    target_moves = " ".join(f"{source}_to_g{rank}" for rank in range(1, 9))

    env = {
        **__import__("os").environ,
        "SOURCE_SQUARES": source,
        "TARGET_MOVES": target_moves,
        "LOOKUP_LIFT_HEIGHT_OVERRIDE": "0.07",
    }

    print(f"Running {source}...")
    subprocess.run(
        [python_exe, "-B", script],
        env=env,
        check=True,
    )

print("Done.")