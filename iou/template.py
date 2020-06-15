def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_intersect_left = max(x1 - 0.5 * w1, x2 - 0.5 * w2)
    x_intersect_right = min(x1 + 0.5 * w1, x2 + 0.5 * w2)
    y_intersect_up = min(y1 + 0.5 * h1, y2 + 0.5 * h2)
    y_intersect_bottom = max(y1 - 0.5 * h1, y2 - 0.5 * h2)
    if x_intersect_right <= x_intersect_left or y_intersect_up <= y_intersect_bottom: # No overlap
        return 0
    I = (x_intersect_right - x_intersect_left) * (y_intersect_up - y_intersect_bottom)
    U = w1 * h1 + w2 * h2 - I # Union = Total Area - I
    return I / U

iou = [IOU(y_test[i], y_pred[i]) for i in range(len(x_test))]


def display(x, box, box_pred):
    index = np.random.randint(0, len(x))
    plt.imshow(x[index].reshape(16, 16).T, cmap='binary', origin='lower', extent=[0, img_size, 0, img_size])
    plt.gca().add_patch(Rectangle((box_pred[index][0], box_pred[index][1]),
                                  box_pred[index][2], box_pred[index][3],
                                  ec='r', fc='none'))
    plt.title("IOU: " + str(iou[index]))
    plt.xticks([])
    plt.yticks([])
    plt.show()


display(x_test, y_test, box_pred)
