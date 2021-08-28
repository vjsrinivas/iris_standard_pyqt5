from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    EVENT_FILE = '../runs/exp11/events.out.tfevents.1624833378.pop-os.7145.0'
    ea = event_accumulator.EventAccumulator(EVENT_FILE)
    ea.Reload()

    # mAP:
    map_ea = ea.Scalars('metrics/mAP_0.5')
    y_axis = [i.value for i in map_ea]
    x_axis = [i for i in range(len(map_ea))]
    fig = plt.figure()
    plt.plot(x_axis, y_axis)
    plt.title('YOLOv4 mAP @ 0.5 IOU')
    plt.xlabel('Iterations')
    plt.ylabel('mAP')
    plt.ylim((0,1))
    plt.savefig('../runs/exp11/map_0_5.png')
    plt.close()

    # 