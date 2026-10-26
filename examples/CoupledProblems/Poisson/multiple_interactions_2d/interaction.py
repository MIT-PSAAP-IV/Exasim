"""Three non-matching interactions in the four-domain chain."""


def interaction():
    # condition 1 is solution continuity; condition 2 is normal-flux balance.
    return [
        dict(domains=[1, 2], type=1,
             interface1=2, condition1=1, fluxmap1=[1],
             interface2=4, condition2=2, fluxmap2=[1]),
        dict(domains=[2, 3], type=1,
             interface1=2, condition1=2, fluxmap1=[1],
             interface2=4, condition2=1, fluxmap2=[1]),
        dict(domains=[3, 4], type=1,
             interface1=2, condition1=1, fluxmap1=[1],
             interface2=4, condition2=2, fluxmap2=[1]),
    ]
