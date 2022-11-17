import elikopy.core
import elikopy.individual_subject_processing
import elikopy.utils
import elikopy.utilsSynb0Disco
import elikopy.registration
try:
    import elikopy.mouse
except ImportError as e:
    elikopy.mouse = None
    print("Mouse module not available: {}".format(e))


try:
    import elikopy.modelSynb0Disco
except ImportError as e:
    elikopy.modelSynb0Disco = None
    print("Synb0Disco module not available: {}".format(e))
