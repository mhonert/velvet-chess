import subprocess
import logging as log

class Engine:
    def __init__(self, engine_cmd):
        self.process = subprocess.Popen([engine_cmd], bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, universal_newlines=True)
        self.is_prepared = False
        self.test_positions = []

    def stop(self):
        log.debug("Stopping engine instance")
        # self.process.communicate("quit\n", timeout=2)
        self.send_command("quit")

        self.process.kill()
        self.process.communicate()

    def send_command(self, cmd):
        # log.debug(">>> " + cmd)
        self.process.stdin.write(cmd + "\n")

    def wait_for_command(self, cmd):
        for line in self.process.stdout:
            line = line.rstrip()
            # log.debug("<<< " + line)
            if cmd in line:
                return line


