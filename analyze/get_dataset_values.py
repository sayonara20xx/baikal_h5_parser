from numpy import arctan2

# функции написаны с упором на НЕ нормированные значения
# ро надо передать разницу соттветствующих координат модуля и каскада
# фи вроде как ясно, это просто угол между каскадом в плоскости и осью
# у тета один катет лежит в плоскости XoY как и задумывалось, другой противолежащий как раз по высоте


def get_rho_det_origin(x_cc, y_cc):
    '''
        расстояние от модуля до точки рождения каскада в плоскости OXY
        в аргументы передаются разности соответствующих координат
        т.е. делаем перенос начала координат в точку детектора +vec(-x_det, -y_det)
    '''
    return (x_cc**2 + y_cc**2)**0.5

def get_phi_det_origin(x_cc, y_cc):
    '''
        Угол между направлением каскада в плоскости OXY и осью OX
        ясно, что можно передать нормированные значения
    '''
    return arctan2(y_cc, x_cc)

def get_theta_det_origin(x_cc, y_cc, z_cc):
    '''
       Угол между направлением каскада и осью OZ
       Противолежащий катет - направление в плоскости OXY
       Прилежащий - направление по Z каскада
       Получаем один из двух накрест лежащих углов, второй - это угол между направлением
       каскада и самой осью, вроде так
    '''
    return arctan2((x_cc**2 + y_cc**2)**0.5, z_cc)

'''
    тут уже мои функции
    Они более сложные, обобщенные и явные, и угол считается через косинус
    их назначение не совпадает с назначениями функций выше
'''

from math import acos, sqrt

def get_rho(det_coords : list, cascade_coords : list):
    '''
        rho is distance between cascade start position
        and detector in XoY plane

        it's calculated using Pythagorean theorem and according coords difference
    '''
    return sqrt(pow(det_coords[0] - cascade_coords[0], 2) + pow(det_coords[1] - cascade_coords[1], 2))


def get_phi(det_coords : list, cascade_coords : list, cascade_vec : list):
    '''
        phi is angle between rho lane and cascade direction in XoY plane

        at first, i'm calculating vector with rho direction and normalize it
        by divide into it's module

        after, i'm calculating angle by formula with 'acos'

        rho (x1, y1)
        cascade (x2, y2)

                                x1 * x2 + y1 * y2
        alpha = acos _______________________________________
                      sqrt(x1^2 + y1^2) * sqrt(x2^2 + y2^2)
    '''
    two_dim_module = get_rho(det_coords, cascade_coords)
    rho_normalized_vector = [(det_coords[0] - cascade_coords[0]) / two_dim_module, 
                             (det_coords[1] - cascade_coords[1]) / two_dim_module]

    temp1 = rho_normalized_vector[0] * cascade_vec[0] + rho_normalized_vector[1] * cascade_vec[1]
    temp2 = get_vec_module_2d(rho_normalized_vector[:2]) * get_vec_module_2d(cascade_vec[:2])
    return acos(temp1 / temp2)


def get_theta(cascade_vec : list):
    '''
        theta is angle between Z and cascade direction
        z direction is (0, 0, 1)
        cascade (x1, y1, z1)
                                 z1
        alpha = acos __________________________
                      sqrt(x1^2 + y1^2 + z1^2)
    '''
    return acos(cascade_vec[2] / get_vec_module_3d(cascade_vec))

def get_vec_module_2d(vec : list):
    return (sqrt(pow(vec[0], 2) + pow(vec[1], 2)))

def get_vec_module_3d(vec : list):
    return (sqrt(pow(vec[0], 2) + pow(vec[1], 2)) + pow(vec[2], 2))