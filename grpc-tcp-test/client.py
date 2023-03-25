import socket

# create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connect to the server
client_socket.connect(('localhost', 9000))

# send value to server
client_value = 30
client_socket.sendall(bytes(str(client_value), 'utf-8'))

print(f'Sent value {client_value} to server')

# receive server value
data = client_socket.recv(1024)
server_value = int(data.decode('utf-8'))

print(f'Received value {server_value} from server')

# find minimum of both values
min_value = min(client_value, server_value)

print(f'Minimum value is {min_value}')

# close the connection
client_socket.close()