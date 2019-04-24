from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
class_name = ["proper", "lying", "left", "right", "leftcross", "rightcross", "leftcross1", "rightcross1"]
#table = [[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]]
table = np.array(all_error_matrix)

font_ = ImageFont.truetype(font="arial.ttf",size=15)
rectangle_size=100
table_x = 150
table_y = 100
text_x = 0.2*rectangle_size
text_y = 0.7*rectangle_size
source_img = Image.new('RGB', (1080, 1080), (255, 255, 255))
draw = ImageDraw.Draw(source_img)

for i in range(table.shape[0]):
	draw.text((table_x-rectangle_size, table_y+i*rectangle_size+text_y), class_name[i], font=font_, fill=(0,0,0))
	draw.text((table_x+i*rectangle_size+text_x, table_y-rectangle_size+text_y), class_name[i], font=font_, fill=(0,0,0))

draw.rectangle(((table_x-1, table_y-1), (table_x+table.shape[0]*rectangle_size+1, table_y+table.shape[0]*rectangle_size+1)), fill=(255,255,255), outline =(0,0,0) )
for i in range(table.shape[0]):
    for j in range(table.shape[1]):
        draw.rectangle(((table_x+j*rectangle_size, table_y+i*rectangle_size), (table_x+(j+1)*rectangle_size, table_y+(i+1)*rectangle_size)), fill=(int(255-table[i][j]*255),int(255-table[i][j]*255),int(255-table[i][j]*255)) )
        if table[i][j]<0.5:
        	draw.text((table_x+text_x+j*rectangle_size, table_y+text_y+i*rectangle_size), str(round(table[i][j]*100,1))+"%", font=font_, fill=(0,0,0))
        else:
        	draw.text((table_x+text_x+j*rectangle_size, table_y+text_y+i*rectangle_size), str(round(table[i][j]*100,1))+"%", font=font_)

source_img.save(table_file, "JPEG")