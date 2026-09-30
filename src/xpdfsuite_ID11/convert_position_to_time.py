

def convert_position_to_time(position, position0=[0,0], x_coude=31, z_retour=1, debit=8.0,
                   section=2.0, long_tri=0.0, section_tri=0.5*5**(1/2), tol=1e-1):
    """
    Convertit la position dans la puce en temps de réaction.
    Temps de réaction (s) au point `position`, depuis la fin du mélangeur.

    t= volume/debit, avec volume cumulé depuis la fin du mélangeur jusqu'à la position `position`.
    l'utilisation du volume plutot que de la surface permet de prendre en compte les variations de section du canal.

    position, position0 : (x, z) en mm (position0 = fin du mélangeur)
    x_coude  : x du palier vertical (mm)
    z_retour : offset en z entre les 2 paliers (mm)
    debit    : débit total en µL/s (défaut 8)
    section  : section du canal en mm² (défaut 2 x 2 mm)
    long_tri, section_tri : zone triangulaire au début (ignorée si long_tri=0, section triangle équilatéral de 2mm de côté par défaut)
    tol      : tolérance (mm) pour décider sur quel palier on se trouve
    """
    x, z = position
    x0, z0 = position0
    

    L1 = abs(x_coude - x0)      # 1er palier horizontal
    L2 = abs(z_retour)     # palier vertical

    if abs(z - (z0 + z_retour)) < tol:          # 3e palier : on revient en sens inverse
        d = L1 + L2 + abs(x - x_coude)
    elif abs(x - x_coude) < tol:         # 2e palier (vertical)
        d = L1 + abs(z - z0)
    else:                                # 1er palier
        d = abs(x - x0)

    # volume cumulé (1 µL = 1 mm³) : zone triangulaire puis zone carrée
    d_tri = min(d, long_tri)
    volume = d_tri * section_tri + (d - d_tri) * section
    #print(f"Position:{position}, time={volume/debit:.2f}s, volume={volume:.2f}mm³, debit={debit:.2f}µL/s")
    return volume / debit