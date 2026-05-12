import math
import copy
from dataset import *
from sklearn.metrics import mean_squared_error
import datetime
import os
import time as time_module


def my_accuracy(true_rul, pred_rul):
    diff = (pred_rul - true_rul)
    for i in range(0, len(diff)):
        if 10 >= diff[i] >= -13:
            diff[i] = 1
        else:
            diff[i] = 0
    return torch.sum(diff)


def compute_s_score(rul_true, rul_pred):
    """
    Both rul_true and rul_pred should be 1D numpy arrays.
    """
    diff = rul_pred - rul_true
    return torch.sum(torch.where(diff < 0, torch.exp(-diff/13)-1, torch.exp(diff/10)-1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def testing_function(model, test_loader, loss_func,  max_rul, device):
    model.eval()
    s_score, score, accuracy, rmse_test = 0, 0, 0, 0

    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        with torch.no_grad():
            test_predict, _ = model.forward(x_test)
            scale_test_predict = (test_predict*max_rul).floor()
            scale_y_test = y_test * max_rul
            rmse_test += loss_func(scale_test_predict, scale_y_test).item()
            s_score += compute_s_score(scale_y_test, scale_test_predict)
            accuracy += my_accuracy(scale_y_test, scale_test_predict)

    test_predict = test_predict.cpu().numpy()
    return rmse_test, s_score, accuracy, test_predict


def test_five_sample(model, test_loader, loss_func, max_rul, num_test_windows, device):
    model.eval()
    score, accuracy, rmse_test = 0, 0, 0
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        with torch.no_grad():
            y_test = (y_test.reshape(-1)) * max_rul
            rul_pred, attention_weight_test = model.forward(x_test)
            rul_pred = rul_pred.reshape(-1)
            rul_pred = rul_pred.detach()
            rul_pred *= max_rul
            preds_for_each_engine = torch.split(rul_pred, num_test_windows)  # [array, array,..]
            y_test = torch.split(y_test, num_test_windows)
            y_test = torch.tensor([item.sum() / len(item) for item in y_test])
            mean_pred_for_each_engine = torch.tensor([item.sum() / len(item) for item in preds_for_each_engine])
            rmse_test += loss_func(y_test, mean_pred_for_each_engine).item()
            score += compute_s_score(y_test, mean_pred_for_each_engine)
            accuracy += my_accuracy(y_test, mean_pred_for_each_engine)

    return rmse_test, score.item()


def validating_function(model, valid_loader, loss_func, max_rul, device):
    rmse_valid = 0
    valid_batch_len = len(valid_loader)
    size = len(valid_loader.dataset)
    score = 0
    with torch.no_grad():
        for x_valid, y_valid in valid_loader:
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            valid_predict, attention_weight_valid = model(x_valid)
            rmse_valid += loss_func(valid_predict, y_valid).item()
            score += compute_s_score(y_valid*125, valid_predict*125)
    rmse_valid /= valid_batch_len
    rmse_valid *= max_rul
    return rmse_valid, score.item(), attention_weight_valid


def train(model_for_train, train_loader, valid_loader, test_loader, N_EPOCH, optimizer, scheduler, loss_train,
          loss_eval, lines_list, patience, max_rul, num_test_windows, device, time, log_path,
          checkpoint_dir=None, checkpoint_prefix=None, log_language='en'):
    train_start_at = datetime.datetime.now()
    train_start_perf = time_module.perf_counter()
    train_batch_length = len(train_loader)
    best_rmse_valid = float("inf")
    best_epoch = 0
    best_metrics = {}
    epochs_without_improve = 0
    best_state_dict = copy.deepcopy(model_for_train.state_dict())

    log_name = os.path.splitext(os.path.basename(log_path))[0]
    # Save a dedicated checkpoint whenever validation RMSE reaches a new best.
    if checkpoint_dir is None:
        best_model_path = os.path.join(os.path.dirname(os.path.dirname(log_path)), "trials",
                                       log_name.replace("train_log_", "best_model_") + ".pkl")
    else:
        checkpoint_prefix = checkpoint_prefix or log_name
        best_model_path = os.path.join(checkpoint_dir, "best_" + checkpoint_prefix + ".pkl")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        
    for epoch in range(1, N_EPOCH + 1):
        model_for_train.train()
        model_for_train.to(device)
        epoch_loss = 0

        for batch, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            outputs = model_for_train(x_train)  # forward pass
            loss = loss_train(outputs[0], y_train)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_for_train.eval()
        rmse_valid, valid_score, attention_weight_valid = validating_function(model_for_train,
                                                                              valid_loader=valid_loader,
                                                                              loss_func=loss_eval,
                                                                              max_rul=max_rul,
                                                                              device=device)
        rmse_test, score_test = test_five_sample(model_for_train, test_loader, loss_eval, max_rul, num_test_windows, device)
        epoch_loss /= train_batch_length
        epoch_loss = math.sqrt((epoch_loss)) * max_rul
        if str(log_language).lower().startswith('zh'):
            content = "第%d轮, 训练损失: %1.4f, 验证集RMSE: %1.4f, 验证集S分数: %1.1f, " \
                      "测试集RMSE: %1.4f, 测试集S分数: %1.1f, 学习率: %1.8f" % \
                      (epoch, epoch_loss, rmse_valid, valid_score, rmse_test, score_test,
                       optimizer.state_dict()['param_groups'][0]['lr'])
        else:
            content = "Epoch: %d, Loss: %1.4f, Rmse_valid: %1.4f, valid_score: %1.1f, " \
                      ", rmse_test: %1.4f, score_test: %1.1f, Learning rate:%1.8f" % \
                      (epoch, epoch_loss, rmse_valid, valid_score, rmse_test, score_test,
                       optimizer.state_dict()['param_groups'][0]['lr'])
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        
        print(content)

        improved = rmse_valid < best_rmse_valid
        if improved:
            best_rmse_valid = rmse_valid
            best_epoch = epoch
            best_metrics = {
                "rmse_valid": rmse_valid,
                "valid_score": valid_score,
                "rmse_test": rmse_test,
                "score_test": score_test,
            }
            epochs_without_improve = 0
            best_state_dict = copy.deepcopy(model_for_train.state_dict())
            torch.save(model_for_train, best_model_path)
            
        else:
            epochs_without_improve += 1
        #早停
        #if patience > 0 and epochs_without_improve >= patience:
        #    stop_msg = ("[Early Stop] Triggered at epoch %d. Best epoch: %d, "
        #                "best_rmse_valid: %1.4f") % (epoch, best_epoch, best_rmse_valid)
        #    with open(log_path, "a", encoding="utf-8") as f:
        #        f.write(stop_msg + "\n")
        #    print(stop_msg)
        #    break

        scheduler.step()

    model_for_train.load_state_dict(best_state_dict)
    if str(log_language).lower().startswith('zh'):
        summary_msg = ("[训练总结] 已恢复第%d轮的最佳模型, 验证集RMSE: %1.4f, "
                       "验证集S分数: %1.1f, 测试集RMSE: %1.4f, 测试集S分数: %1.1f") % (
                           best_epoch,
                           best_metrics.get("rmse_valid", float("nan")),
                           best_metrics.get("valid_score", float("nan")),
                           best_metrics.get("rmse_test", float("nan")),
                           best_metrics.get("score_test", float("nan")),
                       )
    else:
        summary_msg = ("[Train Summary] Restored best model from epoch %d, rmse_valid: %1.4f, "
                       "valid_score: %1.1f, rmse_test: %1.4f, score_test: %1.1f") % (
                           best_epoch,
                           best_metrics.get("rmse_valid", float("nan")),
                           best_metrics.get("valid_score", float("nan")),
                           best_metrics.get("rmse_test", float("nan")),
                           best_metrics.get("score_test", float("nan")),
                       )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_msg + "\n")
    print(summary_msg)

    train_end_at = datetime.datetime.now()
    train_seconds = time_module.perf_counter() - train_start_perf
    train_hms = str(datetime.timedelta(seconds=int(round(train_seconds))))
    if str(log_language).lower().startswith('zh'):
        train_time_msg = (
            "[训练耗时] 开始: %s, 结束: %s, 时长: %s, 总秒数: %.2f" % (
                train_start_at.strftime("%Y-%m-%d %H:%M:%S"),
                train_end_at.strftime("%Y-%m-%d %H:%M:%S"),
                train_hms,
                train_seconds,
            )
        )
    else:
        train_time_msg = (
            "[Train Time] start: %s, end: %s, duration: %s, total_seconds: %.2f" % (
                train_start_at.strftime("%Y-%m-%d %H:%M:%S"),
                train_end_at.strftime("%Y-%m-%d %H:%M:%S"),
                train_hms,
                train_seconds,
            )
        )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(train_time_msg + "\n")
    print(train_time_msg)


def evaluate(model, num_test_windows, test_loader, max_rul, device):
    model.to(device)
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.to(device)
    y_test = (y_test.reshape(-1)) * max_rul
    rul_pred, _ = model.forward(x_test)
    rul_pred = rul_pred.reshape(-1)  # (497,1) -- (497,)
    rul_pred = rul_pred.detach()
    rul_pred *= max_rul
    preds_for_each_engine = torch.split(rul_pred, num_test_windows)
    y_test = torch.split(y_test, num_test_windows)
    y_test = torch.tensor([item.sum() / len(item) for item in y_test])
    y_test, index = y_test.sort(descending=True)
    mean_pred_for_each_engine = torch.tensor([item.sum() / len(item) for item in preds_for_each_engine])
    mean_pred_for_each_engine = torch.index_select(mean_pred_for_each_engine, dim=0, index=index)
    mean_pred_for_each_engine = mean_pred_for_each_engine.floor()
    RMSE = mean_squared_error(y_test, mean_pred_for_each_engine, squared=False).item()
    score = compute_s_score(y_test, mean_pred_for_each_engine)
    return RMSE, score
