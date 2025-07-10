import pandas as pd
import math
import math
import matplotlib.pyplot as plt

def triangle_angles(a, b, c):
    # 根据三边长度计算三个角的度数

    # 使用余弦定理计算各个角的余弦值
    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)

    # 计算各个角的度数
    angle_A = math.degrees(math.acos(cos_A))
    angle_B = math.degrees(math.acos(cos_B))
    angle_C = math.degrees(math.acos(cos_C))

    # 返回各个角的度数
    return angle_A, angle_B, angle_C

# 示例用法


result=[]
data = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0426test1_1.csv")
mag1 = [0,5,0.5]
magdot = [0,-1.2,0]
dis1 = 1.2
dis2 = math.sqrt((mag1[0]-magdot[0])**2+(mag1[1]-magdot[1])**2+(mag1[2]-magdot[2])**2)
# print(dis2)
for i in range(0,len(data)):
    mag2 = [0,0,0]
    mag2[0]=data["x"][i]
    mag2[1]= 5
    mag2[2]= data["z"][i]
    if(abs(mag2[0])<0.5):
        mag2[0] =0
    # if(abs(mag2[1])>6 ):
    #     print("皮肤形变较小")
    #     continue
    
    dis3 = math.sqrt((mag2[0]-0)**2+(mag2[1]-0)**2+(mag2[2]-0)**2)
    # print("==================")
    # print(dis2)
    # print(dis3+dis1)
    # print("==================")
    if(dis1+dis3==dis2):
        print("Angle:0")
    else:
        angle = math.degrees(math.atan(data["x"][i]/(data["y"][i]+1.2)))
        print("Angle C: {:.2f}°".format(angle))
        result.append(angle)
        # angles = triangle_angles(dis1, dis2, dis3)
        # print("Y:",mag2[0])
        # print("Angle C: {:.2f}°".format(angles[2]))


# 创建数据
x = []
print(len(result))
y = [2, 3, 5, 7, 11]

# 创建折线图
plt.plot(result, marker='o')

# 添加标题和标签
# plt.title('Sample Line Plot')
plt.xlabel('Point')
plt.ylabel('Angle')

# 显示图表
plt.grid(True)
plt.show()
 
  
    # else:
    #      dis2 = dis3
    #      print("==================")
    #      print(dis2)
    #      print("==================")
        
    #      print("构不成三角形")
         
    # else:
    # print("Angle A: {:.2f}°".format(angles[0]))
    # print("Angle B: {:.2f}°".format(angles[1]))
   
# a, b, c = 1.2, 7.6, 6.4
# a, b, c = 1.2, 7.6, 5.1
# angles = triangle_angles(a, b, c)
# print("Angle A: {:.2f}°".format(angles[0]))
# print("Angle B: {:.2f}°".format(angles[1]))
# print("Angle C: {:.2f}°".format(angles[2]))

