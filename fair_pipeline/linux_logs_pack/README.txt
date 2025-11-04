Synthetic Linux logs pack (benign + malicious examples)

Files:
- benign.log                         -> Routine maintenance, normal user/admin activity
- auth_bruteforce.log                -> SSH brute-force attempts from a single source, later banned
- privilege_escalation.log           -> www-data attempting sudo/su; repeated auth failures
- persistence_cron_systemd.log       -> Suspicious cron and a custom systemd service
- lateral_movement_ssh.log           -> Internal SSH with key reuse and forced scp
- exfiltration_sftp.log              -> Large outbound SFTP transfer to external IP
- c2_reverse_shell.log               -> Cron beacon to external IP using netcat
- defense_evasion.log                -> auditd/rsyslog stopped, shell history cleared
- impact_ransomware_like.log         -> Encryption/deletion signals and system reboot

Note: These are synthetic for testing parsers and detection pipelines.
