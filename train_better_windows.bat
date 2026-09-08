@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Advanced AI Agent - Better Training
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo   Advanced AI Agent - Better Training
 echo ============================================================
echo.

if not exist "%VENV_PY%" (
  echo [ERROR] .venv was not found.
  echo Run start_windows.bat once first.
  goto :end
)

set "AAA_USER=main"
set /p "AAA_USER=User ID to train [main]: "
if not defined AAA_USER set "AAA_USER=main"

echo.
echo [1/2] Strengthening experience network and cognitive learning ...
"%VENV_PY%" train_better.py --user-id "%AAA_USER%"
if errorlevel 1 (
  echo [ERROR] Learning consolidation failed.
  goto :end
)

echo.
echo This first phase works while you use LM Studio and does NOT modify its GGUF.
echo It retrains the local experience policy, reinforces cognitive memories,
echo and creates a training dataset from your rated experiences.
echo.

choice /C YN /N /M "Also train REAL LoRA weights now? [Y/N]: "
if errorlevel 2 goto :done

set "AAA_BASE_MODEL="
set /p "AAA_BASE_MODEL=Original Hugging Face base model path/repo: "
if not defined AAA_BASE_MODEL (
  echo [WARN] No base model supplied. Skipping LoRA.
  goto :done
)

echo.
echo [2/2] Checking LoRA training dependencies ...
call install_training_windows.bat /nopause
if errorlevel 1 (
  echo [ERROR] Training dependency installation failed.
  goto :end
)

set "CONTINUAL_LEARNING_ENABLED=true"
set "LORA_BASE_MODEL=%AAA_BASE_MODEL%"

echo.
echo Starting versioned LoRA training from positively-rated experiences ...
"%VENV_PY%" train_better.py --user-id "%AAA_USER%" --lora
if errorlevel 1 (
  echo.
  echo [ERROR] LoRA training did not complete.
  echo Read the error above. Common causes: too few positive examples,
  echo insufficient RAM/VRAM, or an incorrect Hugging Face base model.
  goto :end
)

echo.
echo [OK] LoRA adapter trained and versioned.
echo Note: LM Studio is an external inference server. A PEFT LoRA adapter is not
 echo automatically injected into an already-loaded LM Studio GGUF. Use the
 echo transformers_peft backend with the adapter, or convert/load a compatible
 echo adapter/model for your LM Studio workflow.

goto :done

:done
echo.
echo [OK] Training workflow complete.

:end
echo.
pause
