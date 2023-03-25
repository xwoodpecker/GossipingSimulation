import socket

# create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# bind the socket to a public host, and a well-known port
server_socket.bind(('localhost', 9000))

# set the socket to listening mode
server_socket.listen(1)

# get server value
server_value = 50

while(True):
    print('Server is listening on port 9000')

    # wait for a client connection
    conn, addr = server_socket.accept()

    print(f'Connection established from {addr[0]}:{addr[1]}')

    # receive value from client
    data = conn.recv(1024)
    client_value = int(data.decode('utf-8'))

    print(f'Received value {client_value} from client')


    # send server value to client
    conn.sendall(bytes(str(server_value), 'utf-8'))

    print(f'Sent value {server_value} to client')

    # find minimum of both values
    min_value = min(client_value, server_value)

    print(f'Minimum value is {min_value}')

    # close the connection
    conn.close()

    server_value = server_value - 1
