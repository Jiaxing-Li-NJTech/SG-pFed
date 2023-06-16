import torch





def FedAVG_communication(server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params

        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32).cuda()
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models


def communication(server_model, models, client_weights,personal_ration):
    with torch.no_grad():
        # aggregate params

        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32).cuda()
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                personal_temp = (1-personal_ration[client_idx])*models[client_idx].state_dict()[key].cuda()+personal_ration[client_idx]*server_model.state_dict()[key].cuda()
                models[client_idx].state_dict()[key].data.copy_(personal_temp)

    return server_model, models