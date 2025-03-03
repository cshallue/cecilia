from tensorboard.backend.event_processing.event_file_loader import \
    RawEventFileLoader
from tensorboard.compat.proto import event_pb2


def read_events(filename):
  loader = RawEventFileLoader(filename)
  events = []
  for raw_event in loader.Load():
    e = event_pb2.Event.FromString(raw_event)
    events.append(e)
  return events
