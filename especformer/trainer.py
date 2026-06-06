import torch 
import torch.nn as nn


def fit(model, train_dl, test_dl, epochs=300, lr=0.001, patience=100, save_path=None, save_path_stat=None, device='cpu'):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    patience_counter = 0

    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_dl)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_correct = 0
            valid_total = 0

            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

            valid_acc = 100 * valid_correct / valid_total
            valid_loss /= len(test_dl)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)

            print_info = f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}'
            print(print_info)  # print to terminal
            if save_path_stat:
                with open(save_path_stat, 'a') as f:
                    print(print_info, file=f)  # write to file

            if valid_acc > best_acc:
                if save_path:
                    torch.save(model.state_dict(), save_path)
                best_acc = valid_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print('Early stopping')
                break

    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list