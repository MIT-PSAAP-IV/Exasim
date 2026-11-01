import numpy as np


def checkconformalinterface(app):
    """Preserve matching inputs; reject features absent from this release."""
    if isinstance(app, (list, tuple)):
        for item in app:
            checkconformalinterface(item)
        return
    if (np.any(np.asarray(app.get('interfacetype', 0)) != 0) or
            any(np.size(app.get(key, [])) > 0 for key in
                ('interactiontable', 'interactions')) or
            app.get('interactionfile', '')):
        raise ValueError('Nonmatching interfaces and generalized interactions are not available in this source release.')
