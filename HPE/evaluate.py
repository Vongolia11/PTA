import numpy as np

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    This version includes a fix for numerical stability.

    Args
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    # A small number to prevent division by zero
    epsilon = 1e-8

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # Centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # Scale to equal (unit) norm
    # ADDED EPSILON TO PREVENT DIVISION BY ZERO
    X0 = X0 / (normX + epsilon)
    Y0 = Y0 / (normY + epsilon)

    # Optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)

    # Add a check for NaN/Inf in matrix A before SVD
    if np.isnan(A).any() or np.isinf(A).any():
        # If A is still problematic, it indicates a deeper issue with the input data.
        # We can return a default/error state instead of crashing.
        print("Warning: NaN or Inf detected in matrix A before SVD. Returning default values.")
        d = np.inf
        Z = Y # Return original Y
        T = np.eye(X.shape[1]) # Return identity matrix
        b = 1
        c = np.zeros(X.shape[1])
        return d, Z, T, b, c

    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / (normY + epsilon)
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/(ssX + epsilon) - 2 * traceTA * normY / (normX + epsilon)
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c


def error(preds, gts):
    """
    Compute MPJPE and PA-MPJPE given predictions and ground-truths.
    """
    # Ensure inputs are numpy arrays
    preds = np.asarray(preds)
    gts = np.asarray(gts)
    
    N = preds.shape[0]
    num_joints = preds.shape[1]

    # Calculate Mean Per Joint Position Error (MPJPE)
    mpjpe = np.mean(np.sqrt(np.sum(np.square(preds - gts), axis=2)))

    # Calculate Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE)
    pampjpe_per_frame = np.zeros(N)

    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]

        # Use the robust similarity transform function
        d, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        
        # The transformed prediction is Z, so we compare Z with the ground truth
        pampjpe_per_frame[n] = np.mean(np.sqrt(np.sum(np.square(Z - frame_gt), axis=1)))

    pampjpe = np.mean(pampjpe_per_frame)

    return mpjpe, pampjpe
