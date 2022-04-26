def add_network_graph_tensorboard(model, inputs, tb_writer):
    """Unable to implement, pytorch requires input tensor all to be the same type 
    i.e. List[tensor] and List[str] lead to errors
    """    
    tb_writer.add_graph(model, inputs)
    return 

def add_model_weights_as_histogram(model, tb_writer, epoch):
    for name, param in model.named_parameters():
        tb_writer.add_histogram(name.replace('.', '/'), param.data.cpu().abs(), epoch)
    return

def add_pr_curves_to_tensorboard(targets, pred_prob, class_list, tb_writer, epoch, num_classes=22):
       
    for cls_idx in range(num_classes):
        binary_target = targets[:, cls_idx]
        true_prediction_prob = pred_prob[:, cls_idx]
        
        tb_writer.add_pr_curve(class_list[cls_idx], 
                               binary_target, 
                               true_prediction_prob, 
                               global_step=epoch)
        
    return

