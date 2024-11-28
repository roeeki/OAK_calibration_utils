from pynput import keyboard
import time

# callback for key presses, the listener will pass us a key object that
# indicates what key is being pressed
def on_key_press(key):
  print("Key pressed: ", key)
  # so this is a bit of a quirk with pynput,
  # if an alpha-numeric key is pressed the key object will have an attribute
  # char which contains a string with the character, but it will only have
  # this attribute with alpha-numeric, so if a special key is pressed
  # this attribute will not be in the object.
  # so, we end up having to check if the attribute exists with the hasattr
  # function in python, and then check the character
  # here is that in action:
  if hasattr(key, "char") and key.char == "z":
    print("Z PRESSED!")

# same as the key press callback, but for releasing keys
def on_key_release(key):
  print("Key released: ", key)
  # if you need to check for a special key like shift you can
  # do so like this:
  if key == keyboard.Key.shift:
    print("SHIFT KEY RELEASED!")

# create a listener and setup our call backs
keyboard_listener = keyboard.Listener(
    on_press=on_key_press,
    on_release=on_key_release)

# start the listener
print("starting the keyboard listener, will run for 5 seconds...")
keyboard_listener.start()

# let the main thread sleep and then kill the listener
time.sleep(5) feq
print("Time's up, stopping the keyboard listener")
keyboard_listener.stop()
keyboard_listener.join()