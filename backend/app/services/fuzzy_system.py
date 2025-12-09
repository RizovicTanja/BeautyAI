import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random

def fuzzy_makeup_recommendation(undertone: str, face_shape: str):
    # Ulazi
    tone = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'tone')
    shape = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'shape')
    
    # Izlaz
    style = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'style')

    # Tone membership
    tone['topao'] = fuzz.trimf(tone.universe, [0, 0, 0.5])
    tone['neutralan'] = fuzz.trimf(tone.universe, [0.25, 0.5, 0.75])
    tone['hladan'] = fuzz.trimf(tone.universe, [0.5, 1, 1])

    # Shape membership
    shape['round'] = fuzz.trimf(shape.universe, [0, 0, 0.2])
    shape['oval'] = fuzz.trimf(shape.universe, [0.15, 0.35, 0.55])
    shape['square'] = fuzz.trimf(shape.universe, [0.5, 0.7, 0.9])
    shape['heart'] = fuzz.trimf(shape.universe, [0.8, 0.9, 1.0])
    shape['oblong'] = fuzz.trimf(shape.universe, [0.6, 0.75, 0.9])


    # Style membership
    style['natural'] = fuzz.trimf(style.universe, [0, 0.3, 0.6])
    style['glam'] = fuzz.trimf(style.universe, [0.5, 0.8, 1])
    style['evening'] = fuzz.trimf(style.universe, [0.7, 1, 1])

    # Pravila
    rules = [
        # Tone = topao
        ctrl.Rule(tone['topao'] & shape['round'], style['natural']),
        ctrl.Rule(tone['topao'] & shape['oval'], style['glam']),
        ctrl.Rule(tone['topao'] & shape['square'], style['evening']),
        ctrl.Rule(tone['topao'] & shape['heart'], style['glam']),
        ctrl.Rule(tone['topao'] & shape['oblong'], style['natural']),

        # Tone = neutralan
        ctrl.Rule(tone['neutralan'] & shape['round'], style['natural']),
        ctrl.Rule(tone['neutralan'] & shape['oval'], style['glam']),
        ctrl.Rule(tone['neutralan'] & shape['square'], style['evening']),
        ctrl.Rule(tone['neutralan'] & shape['heart'], style['glam']),
        ctrl.Rule(tone['neutralan'] & shape['oblong'], style['natural']),

        # Tone = hladan
        ctrl.Rule(tone['hladan'] & shape['round'], style['glam']),
        ctrl.Rule(tone['hladan'] & shape['oval'], style['evening']),
        ctrl.Rule(tone['hladan'] & shape['square'], style['glam']),
        ctrl.Rule(tone['hladan'] & shape['heart'], style['evening']),
        ctrl.Rule(tone['hladan'] & shape['oblong'], style['evening']),
    ]


    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    # Mapiranje undertone na numeričku vrednost + random varijacija
    tone_map = {"topao": 0.0, "neutralan": 0.5, "hladan": 1.0}
    shape_map = {"round": 0.0, "oval": 0.5, "square": 1.0, "heart": 0.85, "oblong": 0.7 }

    tone_value = tone_map.get(undertone, 0.5) + random.uniform(-0.1, 0.1)
    shape_value = shape_map.get(face_shape, 0.5) + random.uniform(-0.1, 0.1)

    sim.input['tone'] = min(max(tone_value, 0), 1)
    sim.input['shape'] = min(max(shape_value, 0), 1)

    sim.compute()
    result = sim.output['style']

    # Mapiranje rezultata na kategoriju
    if result < 0.4:
        rec = "natural"
    elif result < 0.7:
        rec = "glam"
    else:
        rec = "evening"

    return {"recommended_style": rec}
