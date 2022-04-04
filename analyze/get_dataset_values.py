from numpy import arctan2

# same formulas but detector coords considered as origin

def get_rho_det_origin(x_cc, y_cc):
    '''
        rho is distance between detector (as origin) and cascade start point (x_cc, y_cc are cascade coords)
    '''
    (x_cc**2 + y_cc**2)**0.5

def get_phi_det_origin(x_cc, y_cc):
    '''
        phi is angle between rho lane and cascade direction in XoY plane
    '''
    return arctan2(y_cc, x_cc)


def get_theta_det_origin(x_cc, y_cc, z_cc):
    '''
        theta is angle between cascade Z direction and rho
    '''
    return arctan2(get_rho_det_origin(x_cc, y_cc), z_cc)