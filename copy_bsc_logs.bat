@echo off
:loop
echo Running sync at %TIME%
scp -r nct01026@alogin1.bsc.es:/home/nct/nct01026/ATCI-P1/runs C:/Users/Pedro/Documents/MAI/ATCI/ATCI-P1/
timeout /t 180 /nobreak
goto loop
