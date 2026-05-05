import sys

# 1. Import your custom plugin
import sorcha_custom_activity_cook 

# 2. Force Sorcha to update its internal dictionary to see the new plugin
from sorcha.activity.activity_registration import update_activity_subclasses
update_activity_subclasses()

# 3. Bypass the subprocess dispatcher and call the 'run' module directly
from sorcha_cmdline.run import main

if __name__ == "__main__":
    sys.exit(main())