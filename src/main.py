"""Minimal CLI interpreter with AI fallback."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Optional

from ai.user.user_agent import UserAgent
from ai.user.shell_memory import ShellMemory

@dataclass
class CmdResult:
    code: int
    out: str = ""
    err: str = ""

def try_run_command(argv: Sequence[str]) -> CmdResult:
    """
    1) Ejecutar normal (TTY intacto, sudo funciona)
    2) Si falla, reintentar capturando stderr para diagnóstico
    """

    try:
        result = subprocess.run(argv, check=False)
    except FileNotFoundError:
        return CmdResult(
            code=127,
            err=f"{argv[0]}: command not found\n",
        )
    except PermissionError:
        return CmdResult(
            code=126,
            err=f"{argv[0]}: permission denied\n",
        )

    if result.returncode == 0:
        return CmdResult(code=0)

    # --- reintento diagnóstico (silencioso)
    diag = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )

    return CmdResult(
        code=int(diag.returncode),
        out=diag.stdout or "",
        err=diag.stderr or "",
    )

def is_error(result: CmdResult) -> bool:
    return result.code != 0

def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("Usage: ai <request or command>")
        return 2

    store = ShellMemory()

    result = try_run_command(argv)

    store.append({
        "cmd": argv,
        "result": result.err or result.out,
    })

    # éxito → terminamos
    if result.code != 0:
        context = {
            "user_input": argv,
            "stderr": result.err,
            "history": store.tail_session(10),
        }
        msg = f"""
            user_input: {context['user_input']}
            stderr: {context['stderr']}
            history: {context['history']}
        """
        agent = UserAgent().root()
        response = agent.invoke( msg ).resolve()
        print( "RESOLVED" )
        print( str(response.proposal) )

        # print("\nCommand failed, attempting interpretation...\n", file=sys.stderr)
        # temporal: mostrar diagnóstico
        #if result.err:
        #    print(result.err, file=sys.stderr)

    return result.code

if __name__ == "__main__":
    raise SystemExit(main())
