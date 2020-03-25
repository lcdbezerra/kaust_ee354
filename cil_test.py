#!/usr/bin/env python

#	Execution of Conditional Imitation Learning agent 
#	on Carla v0.9.4
import glob
import os
import sys
import time
import random

try:
    import queue
except ImportError:
    import Queue as queue

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('PythonAPI')[0])
except IndexError:
    pass

import numpy as np
import pygame

from cil_agent import ImitationLearningAgent
import carla


imageSize = (600, 800, 4)
def showImage(display, im):
	im = np.array(im.raw_data).reshape(imageSize)[:,:,:3]
	im = im[:,:,::-1]
	image_surface = pygame.surfarray.make_surface(im.swapaxes(0,1))
	display.blit(image_surface, (0,0))

def get_font():
	fonts = [x for x in pygame.font.get_fonts()]
	default_font = 'ubuntumono'
	font = default_font if default_font in fonts else fonts[0]
	font = pygame.font.match_font(font)
	return pygame.font.Font(font, 14)

def should_quit():
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			return True
		elif event.type == pygame.KEYUP:
			if event.key == pygame.K_ESCAPE:
				return True
	return False


def main():
	actor_list = []
	try:
		
		# Connect to Carla server
		client = carla.Client('localhost', 2000)
		client.set_timeout(10.0)
		world = client.load_world('Town01')
		settings = world.get_settings()
		settings.synchronous_mode = True
		# settings.no_rendering_mode = True
		world.apply_settings(settings)

		blueprint_library = world.get_blueprint_library()
		bp = blueprint_library.find('vehicle.tesla.model3')

		colors = bp.get_attribute('color').recommended_values
		bp.set_attribute('color', '140,0,0')

		transform = random.choice(world.get_map().get_spawn_points())
		# print(transform.location)
		# print(transform.rotation)

		vehicle = world.spawn_actor(bp, transform)
		actor_list.append(vehicle)
		print('created %s' % vehicle.type_id)

		camera_bp = blueprint_library.find("sensor.camera.rgb")
		camera_bp.set_attribute('image_size_x', "%s" % str(imageSize[1]))
		camera_bp.set_attribute('image_size_y', "%s" % str(imageSize[0]))
		camera_bp.set_attribute('fov',			"%s" % str(100.0))
		camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(-15.0, 0, 0))
		camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
		actor_list.append(camera)
		print('created %s' % camera.type_id)

		image_queue = queue.Queue()
		camera.listen(image_queue.put)

		# TURN ON AUTOPILOT FOR TESTING
		# vehicle.set_autopilot(True)
		# print("VEHICLE DIR")
		# print(dir(vehicle))

		# INITIALISE CIL MODEL
		agent = ImitationLearningAgent(vehicle=vehicle, 
									   city_name='Town01', 
									   avoid_stopping=True)


		frame = None

		display = pygame.display.set_mode(
			(imageSize[1], imageSize[0]),
			pygame.HWSURFACE | pygame.DOUBLEBUF)
		# font = get_font()
		clock = pygame.time.Clock()

		while True:
			if should_quit():
				return

			clock.tick()
			world.tick()
			ts = world.wait_for_tick()

			if frame is not None:
				if ts.frame_count != frame + 1:
					logging.warning('frame skip!')
			frame = ts.frame_count

			while True:
				image = image_queue.get()
				if image.frame_number == ts.frame_count:
					break
			im = np.array(image.raw_data).reshape(imageSize)[:,:,:3]
			speed_vec = vehicle.get_velocity()
			forward_speed = np.sqrt(speed_vec.x**2 + speed_vec.y**2 + speed_vec.z**2)

			# RUN MODEL
			# control = agent.run_step(measurements, image, directions=2, target=None) # 2: follow straight
			control = agent._compute_action(im, forward_speed, direction=2)
			vehicle.apply_control(control)

			showImage(display, image)
			pygame.display.flip()





	except KeyboardInterrupt:
		pass
	finally:
		print('\nDisabling synchronous mode...')
		settings = world.get_settings()
		settings.synchronous_mode = False
		world.apply_settings(settings)

		print("Destroying actors...")
		for actor in actor_list:
			actor.destroy()
		print("Done.")

if __name__ == '__main__':
    main()