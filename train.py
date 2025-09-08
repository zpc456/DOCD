import torch
import logging
from tqdm import tqdm, trange
from DOCD_process_mimic4 import get_mimiciv_datasets
from DOCD_process_mimic3 import get_mimiciii_datasets
from utils import *
from Argment import *
from models.DOCD import DOCD
from tensorboardX import SummaryWriter

def reduce_training_set(train_dataset, train_priors, percentage, random_seed=1234):
    if percentage >= 1.0:
        return train_dataset, train_priors

    np.random.seed(random_seed)

    original_size = len(train_dataset)
    reduced_size = int(original_size * percentage)

    indices = np.random.permutation(original_size)[:reduced_size]
    indices = sorted(indices)

    reduced_train_dataset = torch.utils.data.Subset(train_dataset, indices)
    reduced_train_priors = [train_priors[i] for i in indices]

    print(f"Training set size reduced from {original_size} to {reduced_size} ({percentage * 100:.1f}%)")

    return reduced_train_dataset, reduced_train_priors


def prediction_loop(args, model, dataloader, priors_datalaoder, description='Evaluating'):
    batch_size = dataloader.batch_size
    eval_losses = []
    preds = None
    label_ids = None
    model.eval()
    for data, priors_data in tqdm(zip(dataloader, priors_datalaoder), desc=description):
        data, priors_data = prepare_data(data, priors_data, args.device)
        with torch.no_grad():
            outputs = model(data, priors_data)
            loss = outputs[0].mean().item()
            logits = outputs[1]

        labels = data[args.label_key]

        batch_size = data[list(data.keys())[0]].shape[0]
        eval_losses.extend([loss] * batch_size)
        preds = logits if preds is None else nested_concat(preds, logits, dim=0)
        label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)

    if preds is not None:
        preds = nested_numpify(preds)
    if label_ids is not None:
        label_ids = nested_numpify(label_ids)
    metrics = compute_metrics(preds, label_ids)
    metrics['eval_loss'] = round(np.mean(eval_losses), 4)

    for key in list(metrics.keys()):
        if not key.startswith('eval_'):
            metrics['eval_{}'.format(key)] = metrics.pop(key)

    return metrics


def save_model(model, optimizer, scheduler, output_dir, global_step):
    model_save_path = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.save(model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
    torch.save(optimizer.state_dict(), os.path.join(model_save_path, 'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(model_save_path, 'scheduler.pt'))


def main():
    args = ArgParser().parse_args()
    set_seed(args.seed)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_dir = os.path.join(args.output_dir, 'logging')
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    tb_writer = SummaryWriter(log_dir=logging_dir)

    logging.info("Training arguments:")
    output_train_args_file = os.path.join(args.output_dir, 'train_args.txt')
    with open(output_train_args_file, 'w') as writer:
        for arg, value in sorted(vars(args).items()):
            logging.info("%s: %s", arg, value)
            writer.write('{}: {}\n'.format(arg, value))

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)

    dx_leaves_list = []
    dx_ancestors_list = []
    dx_masks_list = []
    for i in range(5, 0, -1):
        leaves, ancestors, masks = build_tree_with_padding(os.path.join('processed_data', args.dataset_name, 'tree.diag.level' + str(i) + '.pk'))
        dx_leaves_list.extend(leaves)
        dx_ancestors_list.extend(ancestors)
        dx_masks_list.extend(masks)
    dx_leaves_list = np.array(dx_leaves_list)
    dx_leaves_list = torch.tensor(dx_leaves_list, dtype=torch.long).to(args.device)
    dx_ancestors_list = np.array(dx_ancestors_list)
    dx_ancestors_list = torch.tensor(dx_ancestors_list, dtype=torch.long).to(args.device)
    dx_masks_list = np.array(dx_masks_list)
    dx_masks_list = torch.tensor(dx_masks_list, dtype=torch.long).to(args.device)

    args.dx_leaves_list = dx_leaves_list
    args.dx_ancestors_list = dx_ancestors_list
    args.dx_masks_list = dx_masks_list

    dict_file = os.path.join('processed_data', args.dataset_name, 'dx_map.p')
    vocab = pickle.load(open(dict_file, 'rb'))
    args.dx_code_size = len(vocab)
    print('dx_code_size:', args.dx_code_size)

    num_tree_nodes = get_rootCode(os.path.join('processed_data', args.dataset_name, 'tree.diag.level2.pk')) + 1
    args.dx_num_tree_nodes = num_tree_nodes
    print('dx_num_tree_nodes:', args.dx_num_tree_nodes)

    proc_leaves_list = []
    proc_ancestors_list = []
    proc_masks_list = []
    for i in range(4, 0, -1):
        leaves, ancestors, masks = build_tree_with_padding(os.path.join('processed_data', args.dataset_name, 'tree.proc.level' + str(i) + '.pk'))
        proc_leaves_list.extend(leaves)
        proc_ancestors_list.extend(ancestors)
        proc_masks_list.extend(masks)
    proc_leaves_list = np.array(proc_leaves_list)
    proc_leaves_list = torch.tensor(proc_leaves_list, dtype=torch.long).to(args.device)
    proc_ancestors_list = np.array(proc_ancestors_list)
    proc_ancestors_list = torch.tensor(proc_ancestors_list, dtype=torch.long).to(args.device)
    proc_masks_list = np.array(proc_masks_list)
    proc_masks_list = torch.tensor(proc_masks_list, dtype=torch.long).to(args.device)

    args.proc_leaves_list = proc_leaves_list
    args.proc_ancestors_list = proc_ancestors_list
    args.proc_masks_list = proc_masks_list

    dict_file = os.path.join('processed_data', args.dataset_name, 'proc_map.p')
    vocab = pickle.load(open(dict_file, 'rb'))
    args.proc_code_size = len(vocab)
    print('proc_code_size:', args.proc_code_size)

    num_tree_nodes = get_rootCode(os.path.join('processed_data', args.dataset_name, 'tree.proc.level3.pk')) + 1
    args.proc_num_tree_nodes = num_tree_nodes
    print('proc_num_tree_nodes:', args.proc_num_tree_nodes)

    datasetName = args.dataset_name
    dataset_function_name = f"get_{datasetName}_datasets"
    datasets_function = globals()[dataset_function_name]
    datasets, prior_guides = datasets_function(args.data_dir, fold=args.fold, flag=args.label_key)
    train_dataset, eval_dataset, test_dataset = datasets
    train_priors, eval_priors, test_priors = prior_guides

    if args.train_percentage < 1.0:
        train_dataset, train_priors = reduce_training_set(train_dataset, train_priors, args.train_percentage, random_seed=1234)

    train_priors_dataset = CustomizedDataset(train_priors)
    eval_priors_dataset = CustomizedDataset(eval_priors)
    test_priors_dataset = CustomizedDataset(test_priors)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    train_priors_dataloader = DataLoader(train_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    eval_priors_dataloader = DataLoader(eval_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    test_priors_dataloader = DataLoader(test_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)

    modelName = args.model_name
    model_class = globals()[modelName]
    model = model_class(args)
    model = model.to(args.device)

    if args.do_train:
        num_update_steps_per_epoch = len(train_dataloader)

        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(args.max_steps % num_update_steps_per_epoch > 0)

        else:
            max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
            num_train_epochs = args.num_train_epochs

        num_train_epochs = int(np.ceil(num_train_epochs))

        args.eval_steps = num_update_steps_per_epoch // 2

        optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
        warmup_steps = max_steps // (1 / args.warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps=max_steps)

        logger.info('***** Running Training *****')
        logger.info(' Num examples = {}'.format(len(train_dataloader.dataset)))
        logger.info(' Num epochs = {}'.format(num_train_epochs))
        logger.info(' Train batch size = {}'.format(args.batch_size))
        logger.info(' Total optimization steps = {}'.format(max_steps))

        epochs_trained = 0
        global_step = 0
        tr_loss = torch.tensor(0.0).to(args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        train_pbar = trange(epochs_trained, num_train_epochs, desc='Epoch')

        def save_model(model, output_dir):
            model_save_path = os.path.join(output_dir, 'best_model.bin')
            torch.save(model, model_save_path)


        best_val_loss = float('inf')
        best_step = 0

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_pbar = tqdm(train_dataloader, desc='Iteration')
            for data, priors_data in zip(train_dataloader, train_priors_dataloader):
                model.train()
                data, priors_data = prepare_data(data, priors_data, args.device)

                outputs = model(data, priors_data)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()

                tr_loss += loss.detach()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    tr_loss_scalar = tr_loss.item()
                    logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / args.logging_steps
                    logs['learning_rate'] = scheduler.get_last_lr()[0]
                    logging_loss_scalar = tr_loss_scalar
                    if tb_writer:
                        for k, v in logs.items():
                            if isinstance(v, (int, float)):
                                tb_writer.add_scalar(k, v, global_step)
                        tb_writer.flush()
                    output = {**logs, **{"step": global_step}}

                if (args.eval_steps > 0 and global_step % args.eval_steps == 0):
                    metrics = prediction_loop(args, model, eval_dataloader, eval_priors_dataloader)
                    val_loss = metrics['eval_loss']
                    logger.info('**** Checkpoint Eval Results ****')
                    for key, value in metrics.items():
                        logger.info('{} = {}'.format(key, value))
                        tb_writer.add_scalar(key, value, global_step)
                    if args.output_model:
                        if val_loss < best_val_loss:
                            logger.info("Saving best model at step {}".format(global_step))
                            save_model(model, args.output_dir)
                            best_val_loss = val_loss
                            best_step = global_step

                epoch_pbar.update(1)
                if global_step >= max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            if global_step >= max_steps:
                break

        train_pbar.close()
        if tb_writer:
            tb_writer.close()

        logging.info('\n\nTraining completed')

    eval_results = {}
    if args.do_eval:
        logger.info('*** Evaluate ***')
        logger.info(' Num examples = {}'.format(len(eval_dataloader.dataset)))
        eval_result = prediction_loop(args, model, eval_dataloader, eval_priors_dataloader)
        output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
        with open(output_eval_file, 'w') as writer:
            logger.info('*** Eval Results ***')
            for key, value in eval_result.items():
                logger.info("{} = {}".format(key, value))
                writer.write('{} = {}\n'.format(key, value))
        eval_results.update(eval_result)

    if args.do_test:
        logging.info('*** Test ***')
        test_result = prediction_loop(args, model, test_dataloader, test_priors_dataloader, description='Testing')
        output_test_file = os.path.join(args.output_dir, 'test_results.txt')
        with open(output_test_file, 'w') as writer:
            logger.info('**** Test results ****')
            for key, value in test_result.items():
                logger.info('{} = {}'.format(key, value))
                writer.write('{} = {}\n'.format(key, value))
        eval_results.update(test_result)


def get_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        shape = param.shape
        param_size = 1
        for dim in shape:
            param_size *= dim
        print("name:", name, "shape:", shape, "param_size:", param_size)
        total_params += param_size
    print("total_params:", total_params)


if __name__ == "__main__":
    main()
