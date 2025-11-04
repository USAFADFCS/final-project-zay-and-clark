commands
*Only tested and working on Ai-Capstone server. Running locally will require downloading mistral-7b, DeBERTa-v3 zeroshot-v2.0 and will require significant RAM and GPU resources.

1) python main.py full_pipeline linux_logs_pack/benign.log ./kb report.html
2) python main.py full_pipeline linux_logs_pack/auth_bruteforce.log ./kb report.html
3) python main.py full_pipeline linux_logs_pack/c2_reverse_shell.log ./kb report.html
4) python main.py full_pipeline linux_logs_pack/defense_evasion.log ./kb report.html
5) python main.py full_pipeline linux_logs_pack/exfiltration_sftp.log ./kb report.html
6) python main.py full_pipeline linux_logs_pack/impact_ransomware_like.log ./kb report.html
