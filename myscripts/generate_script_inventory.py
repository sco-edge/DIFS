#!/usr/bin/env bash
# =============================================================================
# generate_script_inventory.py — Auto-generate script documentation for a project
# =============================================================================
# Usage:
#   python3 generate_script_inventory.py <project_root> <output_tex_file>
#
# This scans <project_root> for all .py and .sh files, generates a LaTeX
# document with two sections: Shell Scripts and Python Scripts.
# Run periodically or after adding new scripts to keep docs current.
# =============================================================================

import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def get_script_description(filepath, project_root):
    """Extract a short description from the file's docstring, comments, or first meaningful lines."""
    try:
        with open(filepath, 'r', errors='replace') as f:
            content = f.read(4000)  # Read first 4KB
    except Exception:
        return "No description available."

    lines = content.split('\n')
    description = ""

    # For Python: try docstring
    if filepath.endswith('.py'):
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = True
                    # Single-line docstring
                    if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                        description = stripped.strip('"""').strip("'''").strip()
                        break
                    # Multi-line docstring start
                    after = stripped[3:].strip()
                    if after:
                        description = after
                    continue
                elif stripped.startswith('#'):
                    # Comment line — might be a header/description
                    candidate = stripped.lstrip('#').strip()
                    if candidate and len(candidate) > 5 and not candidate.startswith('!'):
                        if not description:
                            description = candidate
                        break
                elif stripped and not stripped.startswith('import') and not stripped.startswith('from'):
                    break
            else:
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    desc_line = stripped[:-3].strip()
                    if desc_line:
                        description = (description + " " + desc_line).strip() if description else desc_line
                    in_docstring = False
                    break
                elif stripped:
                    description = (description + " " + stripped).strip() if description else stripped

    # For Shell: try shebang comment block or first # comment lines
    elif filepath.endswith('.sh'):
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#!'):
                continue
            if stripped.startswith('#'):
                candidate = stripped.lstrip('#').strip()
                if candidate and len(candidate) > 5:
                    # Skip common header lines
                    if any(skip in candidate.lower() for skip in ['===', '---', 'usage:']):
                        if 'usage:' in candidate.lower():
                            description = candidate
                            break
                        continue
                    if not description:
                        description = candidate
                    break
            elif stripped:
                break

    # Clean up
    description = description.replace('\\', '\\textbackslash{}')
    description = description.replace('&', '\\&')
    description = description.replace('%', '\\%')
    description = description.replace('#', '\\#')
    description = description.replace('_', '\\_')
    description = description.replace('~', '\\textasciitilde{}')

    if not description:
        description = "No description available."
    return description[:200]  # Truncate long descriptions


def relative_path(filepath, project_root):
    """Get path relative to project root."""
    return os.path.relpath(filepath, project_root)


def get_directory(filepath, project_root):
    """Get the directory group for a script."""
    rel = relative_path(filepath, project_root)
    parts = Path(rel).parts

    if len(parts) == 1:
        return "Root Directory"

    top_dir = parts[0]

    # Map to readable group names
    group_map = {
        'scripts': 'scripts/',
        'myscripts': 'myscripts/',
        'src': 'src/',
        'protos': 'protos/',
    }

    if top_dir in group_map:
        return group_map[top_dir]

    if top_dir == 'DiffServ_0112_1730_coco2':
        if len(parts) >= 4:
            inner = parts[2]  # 0112_1730_coco level
            sub = parts[3] if len(parts) > 3 else ''
            if sub == 'protos':
                return 'DiffServ protos/'
            elif sub == 'src':
                if len(parts) > 4:
                    deeper = parts[4]
                    if deeper == '1':
                        return 'DiffServ src/1/'
                    elif deeper == 'plots':
                        return 'DiffServ src/plots/'
                    elif deeper == 'experiments':
                        return 'DiffServ src/experiments/'
                    elif deeper == 'scripts':
                        return 'DiffServ src/scripts/'
                    else:
                        return 'DiffServ src/'
                return 'DiffServ src/'
            elif sub == 'src' and len(parts) > 4:
                return f'DiffServ {sub}/'
            else:
                return 'DiffServ root/'
        return 'DiffServ/'

    if top_dir in ('build', 'cmake', 'dockerfiles', 'data', 'include', 'my_config',
                    'logs', 'error_and_recent_command_logs', 'screeshots_pnb',
                    'Notes', 'utils', 'thirdparty', 'mymodel_directory'):
        return None  # Skip non-script directories

    return top_dir + '/'


def scan_scripts(project_root):
    """Scan project for all .py and .sh files, returning structured inventory."""
    python_scripts = []
    shell_scripts = []
    skip_dirs = {'.git', '__pycache__', 'node_modules', 'build', '.git-annex',
                 'cmake', 'dockerfiles', 'data', 'include', 'my_config',
                 'logs', 'error_and_recent_command_logs', 'screeshots_pnb',
                 'Notes', 'utils', 'thirdparty', 'mymodel_directory',
                 'results', 'environment_snapshot', 'WorkersLogs',
                 'metrics', 'ClientReceived', 'ClientSent',
                 'plots', 'prompts', 'configs'}

    for root, dirs, files in os.walk(project_root):
        # Skip hidden and non-script directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]

        for f in sorted(files):
            if f.endswith('.py~') or f.endswith('.sh~') or f.endswith('.bak'):
                continue
            filepath = os.path.join(root, f)
            if f.endswith('.py'):
                python_scripts.append(filepath)
            elif f.endswith('.sh'):
                shell_scripts.append(filepath)

    return shell_scripts, python_scripts


def generate_tex(project_root, output_file, project_name="DIFS"):
    """Generate LaTeX script inventory document."""
    project_root = os.path.abspath(project_root)
    shell_scripts, python_scripts = scan_scripts(project_root)

    # Group scripts
    shell_groups = defaultdict(list)
    python_groups = defaultdict(list)

    for f in shell_scripts:
        group = get_directory(f, project_root)
        if group:
            shell_groups[group].append(f)

    for f in python_scripts:
        group = get_directory(f, project_root)
        if group:
            python_groups[group].append(f)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build LaTeX
    lines = []
    lines.append(r"\chapter{Script Inventory}")
    lines.append(r"\label{ch:script_inventory}")
    lines.append("")
    lines.append(f"Auto-generated on {now}. Contains {len(shell_scripts)} shell scripts and {len(python_scripts)} Python scripts.")
    lines.append(f"Regenerate with: \\texttt{{python3 myscripts/generate\\_script\\_inventory.py .. <output\\_file>}}")
    lines.append("")

    # ---- SHELL SCRIPTS SECTION ----
    lines.append(r"\section{Shell Scripts (.sh)}")
    lines.append(f"This project contains {len(shell_scripts)} shell scripts.")
    lines.append("")

    shell_order = [
        "Root Directory", "scripts/", "myscripts/", "src/",
        "DiffServ root/", "DiffServ protos/", "DiffServ src/",
        "DiffServ src/1/", "DiffServ src/plots/",
        "DiffServ src/experiments/", "DiffServ src/scripts/",
    ]
    all_shell_groups = list(shell_groups.keys())
    # Sort: ordered ones first, then any remaining
    ordered = [g for g in shell_order if g in all_shell_groups]
    remaining = [g for g in all_shell_groups if g not in ordered]
    sorted_shell_groups = ordered + sorted(remaining)

    for group in sorted_shell_groups:
        scripts = shell_groups[group]
        label = group.replace('/', '_').replace(' ', '_')
        lines.append(r"\subsection{" + group + "}")
        lines.append("")
        lines.append(r"\begin{longtable}{p{0.5cm} p{7cm} p{8.5cm}}")
        lines.append(r"\caption{" + group + r" Shell Scripts} \\")
        lines.append(r"\label{tab:shell_" + label + r"} \\")
        lines.append(r"\toprule")
        lines.append(r"\textbf{\#} & \textbf{Relative Path} & \textbf{Description} \\")
        lines.append(r"\midrule")
        lines.append(r"\endfirsthead")
        lines.append(r"\midrule")
        lines.append(r"\textbf{\#} & \textbf{Relative Path} & \textbf{Description} \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")

        for i, f in enumerate(scripts, 1):
            desc = get_script_description(f, project_root)
            rel = relative_path(f, project_root)
            rel_esc = rel.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
            lines.append(f"{i}. \\texttt{{{rel_esc}}} --- {desc} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
        lines.append("")

    # ---- PYTHON SCRIPTS SECTION ----
    lines.append(r"\section{Python Scripts (.py)}")
    lines.append(f"This project contains {len(python_scripts)} Python scripts.")
    lines.append("")

    python_order = [
        "Root Directory", "protos/", "scripts/", "myscripts/", "src/",
        "src/containers/", "src/containers/diffusion/",
        "src/containers/gnmt-nvpy/", "src/containers/inferentia/",
        "src/profiler/", "src/metadata-store/", "src/worker/",
        "DiffServ root/", "DiffServ protos/", "DiffServ src/",
        "DiffServ src/1/", "DiffServ src/plots/",
        "DiffServ src/experiments/", "DiffServ src/scripts/",
    ]
    all_python_groups = list(python_groups.keys())
    ordered = [g for g in python_order if g in all_python_groups]
    remaining = [g for g in all_python_groups if g not in ordered]
    sorted_python_groups = ordered + sorted(remaining)

    for group in sorted_python_groups:
        scripts = python_groups[group]
        label = group.replace('/', '_').replace(' ', '_')
        lines.append(r"\subsection{" + group + "}")
        lines.append("")
        lines.append(r"\begin{longtable}{p{0.5cm} p{7cm} p{8.5cm}}")
        lines.append(r"\caption{" + group + r" Python Scripts} \\")
        lines.append(r"\label{tab:python_" + label + r"} \\")
        lines.append(r"\toprule")
        lines.append(r"\textbf{\#} & \textbf{Relative Path} & \textbf{Description} \\")
        lines.append(r"\midrule")
        lines.append(r"\endfirsthead")
        lines.append(r"\midrule")
        lines.append(r"\textbf{\#} & \textbf{Relative Path} & \textbf{Description} \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")

        for i, f in enumerate(scripts, 1):
            desc = get_script_description(f, project_root)
            rel = relative_path(f, project_root)
            rel_esc = rel.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
            lines.append(f"{i}. \\texttt{{{rel_esc}}} --- {desc} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
        lines.append("")

    # Write output
    tex_content = '\n'.join(lines)
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(tex_content)

    print(f"Generated: {output_file}")
    print(f"  Shell scripts: {len(shell_scripts)}")
    print(f"  Python scripts: {len(python_scripts)}")
    print(f"  Total: {len(shell_scripts) + len(python_scripts)}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <project_root> <output_tex_file>")
        print(f"Example: {sys.argv[0]} . results/model_benchmark/Reports/ScriptInventory.tex")
        sys.exit(1)

    project_root = sys.argv[1]
    output_file = sys.argv[2]

    # Derive project name from root
    project_name = os.path.basename(os.path.abspath(project_root))
    generate_tex(project_root, output_file, project_name)
