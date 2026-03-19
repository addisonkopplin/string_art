import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_original = np.array(Image.open('einstein.jpg').convert('L'))/255
upscale_factor = 1200//np.max(image_original.shape)
image_original = np.kron(image_original, np.ones((upscale_factor, upscale_factor)))
height, width = image_original.shape
r = int(max([width,height])/2)+1
image = np.ones([2*r,2*r])*image_original[0,0]


dCenterX = (2*r-width)//2
dCenterY = (2*r-height)//2
image[dCenterY:dCenterY+height,dCenterX:dCenterX+width] = image_original

#image = image_original[height//2-r:height//2+r,width//2-r:width//2+r]
image_copy = np.copy(image)
blank = np.ones(image.shape)
# blank[0,0] = 0

image_size = image.shape

# Specify circle parameters
center = (r, r)
radius = r

# Generate a grid of coordinates
x, y = np.ogrid[:image_size[0], :image_size[1]]

# Create a mask for the circle
mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

# Set the pixels inside the circle to 1
image_copy[mask == 0] = 1

# theta = np.radians(np.linspace(0,360,361))
# cirX = np.cos(theta)*r
# cirY = np.sin(theta)*r

n_nails = 100
dtheta = 360/n_nails
nail_theta = np.radians(np.arange(0,360,dtheta))
nails = np.arange(n_nails)

nailX = np.cos(nail_theta)*r
nailY = np.sin(nail_theta)*r

plt.figure()
plt.imshow(image_copy,extent = (-(r-1), (r-1), -(r-1), (r-1)), cmap = 'gray')
plt.scatter(nailX, nailY,s=0.5)
plt.show()

lines = np.copy(blank)
n_lines = 350
current_nail = 0
previous_nail = 0
line_color = 0.2
n_skip = 0

err_his = np.zeros(n_lines)

nail_order = np.zeros(n_lines+1).astype(int)

for i in range(n_lines):
    total_err = np.sqrt(((image_copy-lines)**2).mean())
    err_his[i] = total_err
    if i>300 and total_err > np.mean(err_his[i-300:i]):
        print('break')
        break

    #print('-----------new line---------------')
    print('# of lines = ', i, end = '\r')

    #print(previous_nail, current_nail)

    current_err = 1
    nails_skip = np.concatenate((np.arange(current_nail - n_skip, current_nail), np.arange(current_nail, current_nail + n_skip+1)))%n_nails
    # print(nails_skip)
    for j in (np.delete(nails, np.append(nails_skip, [previous_nail]))):

        Ax = nailX[current_nail].astype(int)
        Ay = nailY[current_nail].astype(int)

        Bx = nailX[j].astype(int)
        By = nailY[j].astype(int)

        x_start = np.min([Ax,Bx])
        x_end = np.max([Ax,Bx])
        dx = x_end - x_start

        y_start = np.min([Ay,By])
        y_end = np.max([Ay,By])
        dy = y_end - y_start

        line = np.copy(blank)
        s=1

        if dx>dy:
            for k in range(x_start,x_end,1):
                x0 = k
                y0 = (Ay-By)/(Ax-Bx)*(x0 - Ax) + Ay 
                row = int((r-1 - y0))
                col = int((x0 + r-1))
                #line[row-s:row+s,col-s:col+s] -= 80
                line[row,col] = line_color

        else:
            for k in range(y_start,y_end,1):
                y0 = k
                x0 = (Ax-Bx)/(Ay-By)*(y0 - Ay) + Ax
                row = int((r-1 - y0))
                col = int((x0 + r-1))
                #line[row-s:row+s,col-s:col+s] -= 80 
                line[row,col] = line_color

        err = (image[line==line_color]).mean()

        if err < current_err:
            current_err = err
            j_best = j
            line_best = line

    previous_nail = current_nail
    current_nail = j_best
    nail_order[i+1] = current_nail
    lines[line_best == line_color] = line_color 
    image[np.where(line_best == line_color)] = 1



comp = np.copy(image)
# image[lines < 1] = lines[lines < 1]


n = np.argmin(err_his)
plt.figure()
# plt.imshow(lines,extent = (-(r-1), (r-1), -(r-1), (r-1)), cmap = 'gray')
# plt.scatter(nailX, nailY,s=1)
plt.axis('equal')
plt.plot(nailX[nail_order[:n]]*10, nailY[nail_order[:n]]*10, color='black', alpha=0.8, linewidth=0.1)
plt.figure(figsize=(2, 2))
plt.show()

plt.figure()
plt.plot(err_his)
plt.show()

print(nail_order)


np.save('nail_order.npy',nail_order)

