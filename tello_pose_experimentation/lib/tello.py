import socket

class Tello:

    def __init__(self):
        self.port = 8889
        self.addr = ('192.168.10.1', self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, message):
        try:
            self.sock.sendto(message.encode(), self.addr)
            print("Sending message: " + message)
        except Exception as e:
            print("Error sending: " + str(e))

    def receive(self):
        return


    def close(self):
        self.sock.close()
