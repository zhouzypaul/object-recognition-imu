def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0
    I = w_intersection * h_intersection
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
