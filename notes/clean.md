### How to clean up all stale Gazebo processes
```bash
pkill -9 -f "ign gazebo"
```

### Verify everything is actually dead
```bash
ps aux | grep -i gazebo
```
Should show nothing except the `grep` command itself matching its own name.

### If a specific process refuses to die (manual kill by PID)
1. Find its PID from the `ps aux` output (second column)
2. Run:
```bash
kill -9 <PID>
```
(Replace `<PID>` with the actual number — e.g. `kill -9 45673`.
Do NOT run `kill -9 <PID>` literally with the placeholder text —
substitute the real number.)
