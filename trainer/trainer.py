import torch


def train_one_epoch(
        model,
        train_dataloader,
        loss_log,
        optimizer,
        device,
        lr_scheduler=None,
):
    model = model.to(device)
    model.train()

    for x_batch, y_batch in train_dataloader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_batch_ = model(x_batch)
        loss = loss_log(y_batch_, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()


@torch.no_grad()
def test(
        model,
        test_dataloader,
        inference_func,
        metric_log,
        device
):
    model = model.to(device)
    model.eval()

    for x_batch, y_batch in test_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_batch_ = inference_func(x_batch, model)
        metric_log(y_batch_, y_batch)



